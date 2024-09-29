package com.example.safeT.kickboard;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/ai")
public class AIController {

    @Autowired
    private AIService aiService;

    // 얼굴 인식 요청 처리
    @GetMapping("/face-recognition")
    public CompletableFuture<ResponseEntity<String>> faceRecognition(@RequestParam String userId) {
        return aiService.sendUserIdToPython(userId)
                .thenApply(result -> ResponseEntity.ok(result))
                .exceptionally(e -> ResponseEntity.status(500).body("An error occurred: " + e.getMessage()));
    }

    // 헬멧 감지 요청 처리
    @GetMapping("/helmet-detection")
    public CompletableFuture<ResponseEntity<String>> helmetDetection(@RequestParam String userId) {
        return aiService.detectHelmet(userId)
                .thenApply(result -> ResponseEntity.ok(result))
                .exceptionally(e -> ResponseEntity.status(500).body("An error occurred: " + e.getMessage()));
    }

    // 2인 이상 탑승 감지 요청 처리
    @GetMapping("/people-detection")
    public CompletableFuture<ResponseEntity<String>> peopleDetection(@RequestParam String userId) {
        return aiService.detectPeople(userId)
                .thenApply(result -> ResponseEntity.ok(result))
                .exceptionally(e -> ResponseEntity.status(500).body("An error occurred: " + e.getMessage()));
    }

    // 횡단보도 감지 요청 처리
    @GetMapping("/crosswalk-detection")
    public CompletableFuture<ResponseEntity<String>> crosswalkDetection(@RequestParam String userId) {
        return aiService.detectCrosswalk(userId)
                .thenApply(result -> ResponseEntity.ok(result))
                .exceptionally(e -> ResponseEntity.status(500).body("An error occurred: " + e.getMessage()));
    }
}
