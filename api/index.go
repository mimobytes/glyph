package handler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"google.golang.org/genai"
)

const defaultModel = "gemini-2.5-flash-lite"

var (
	once sync.Once
	mux  *http.ServeMux
)

type TranslateRequest struct {
	Text           string `json:"text"`
	TargetLanguage string `json:"target_language"`
	To             string `json:"to"`
}

func getAPIKey() string {
	if key := os.Getenv("GEMINI_API_KEY"); key != "" {
		return key
	}
	return "AQ.Ab8RN6JZyiS0GSMh5vf56qfRjmt9lkLkdINKM5u6f6_ndSkQdw"
}

func getMIMEType(filename string, fileBytes []byte) string {
	ext := filepath.Ext(filename)
	if detectedMIME := mime.TypeByExtension(ext); detectedMIME != "" {
		if parsed, _, err := mime.ParseMediaType(detectedMIME); err == nil {
			return parsed
		}
	}
	if bytes.HasPrefix(fileBytes, []byte{0xFF, 0xD8, 0xFF}) {
		return "image/jpeg"
	}
	if bytes.HasPrefix(fileBytes, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'}) {
		return "image/png"
	}
	if bytes.HasPrefix(fileBytes, []byte("GIF87a")) || bytes.HasPrefix(fileBytes, []byte("GIF89a")) {
		return "image/gif"
	}
	if bytes.HasPrefix(fileBytes, []byte("RIFF")) && len(fileBytes) >= 12 && bytes.Contains(fileBytes[:12], []byte("WEBP")) {
		return "image/webp"
	}
	return "image/png"
}

func extractJSON(raw string) string {
	re := regexp.MustCompile(`(?s)\{.*\}`)
	if match := re.FindString(raw); match != "" {
		return match
	}
	return raw
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func handleHome(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("Welcome! Translation API is working."))
}

func handleStatus(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  getAPIKey(),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"api_key_valid": false,
			"status":        "error",
			"model":         defaultModel,
			"error":         err.Error(),
		})
		return
	}

	contents := []*genai.Content{
		genai.NewContentFromParts([]*genai.Part{genai.NewPartFromText("ping")}, genai.RoleUser),
	}

	_, err = client.Models.GenerateContent(ctx, defaultModel, contents, nil)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"api_key_valid": false,
			"status":        "error",
			"model":         defaultModel,
			"error":         err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"api_key_valid": true,
		"status":        "operational",
		"model":         defaultModel,
		"capabilities":  []string{"image-translation", "text-translation"},
	})
}

func handleTranslate(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	contentType := r.Header.Get("Content-Type")

	var (
		targetLang string
		parts      []*genai.Part
	)

	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Failed to parse multipart form"})
			return
		}

		targetLang = r.FormValue("target_language")
		if targetLang == "" {
			targetLang = r.FormValue("to")
		}

		if textInput := r.FormValue("text"); textInput != "" {
			parts = append(parts, genai.NewPartFromText(fmt.Sprintf("Text to translate: %s", textInput)))
		}

		file, header, err := r.FormFile("image")
		if err == nil {
			defer file.Close()
			fileBytes, err := io.ReadAll(file)
			if err == nil && len(fileBytes) > 0 {
				mimeType := getMIMEType(header.Filename, fileBytes)
				parts = append(parts, genai.NewPartFromBytes(fileBytes, mimeType))
			}
		}
	} else {
		var req TranslateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid JSON body"})
			return
		}

		targetLang = req.TargetLanguage
		if targetLang == "" {
			targetLang = req.To
		}

		if req.Text != "" {
			parts = append(parts, genai.NewPartFromText(fmt.Sprintf("Text to translate: %s", req.Text)))
		}
	}

	if targetLang == "" {
		targetLang = "English"
	}

	if len(parts) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "No text or image provided for translation"})
		return
	}

	instruction := fmt.Sprintf(`You are a professional translator and linguistic analyzer.
Translate all text extracted from the provided input (image or text) accurately into %s.
Identify the source language, script, and linguistic characteristics.

Return only valid JSON with this structure:
{
  "detected_language": "",
  "detected_script": "",
  "target_language": "%s",
  "original_text": "",
  "translated_text": "",
  "confidence": 0.0,
  "additional_info": {
    "linguistic_family": "",
    "cultural_notes": ""
  }
}`, targetLang, targetLang)

	promptParts := append([]*genai.Part{genai.NewPartFromText(instruction)}, parts...)
	contents := []*genai.Content{
		genai.NewContentFromParts(promptParts, genai.RoleUser),
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  getAPIKey(),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	temp := float32(0.1)
	config := &genai.GenerateContentConfig{
		Temperature:      &temp,
		ResponseMIMEType: "application/json",
	}

	resp, err := client.Models.GenerateContent(ctx, defaultModel, contents, config)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	var result map[string]any
	rawText := resp.Text()
	if err := json.Unmarshal([]byte(rawText), &result); err != nil {
		extracted := extractJSON(rawText)
		if err := json.Unmarshal([]byte(extracted), &result); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "Failed to parse translation result"})
			return
		}
	}

	writeJSON(w, http.StatusOK, result)
}

func initRouter() {
	mux = http.NewServeMux()
	mux.HandleFunc("GET /home", handleHome)
	mux.HandleFunc("GET /status", handleStatus)
	mux.HandleFunc("POST /translate", handleTranslate)
	mux.HandleFunc("GET /", handleHome)
}

func Handler(w http.ResponseWriter, r *http.Request) {
	once.Do(initRouter)
	mux.ServeHTTP(w, r)
}
