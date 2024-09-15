package com.example.safeT.kickboard;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/ai")
public class AIController {

    @Autowired
    private AIService aiService;

    // 얼굴 인식 요청 처리
    @PostMapping("/face-recognition")
    public ResponseEntity<String> faceRecognition(@RequestParam String userId) {
        try {
            String result = aiService.sendUserIdToPython(userId);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("An error occurred: " + e.getMessage());
        }
    }

    // 헬멧 감지 요청 처리
    @PostMapping("/helmet-detection")
    public ResponseEntity<String> helmetDetection(@RequestParam String userId) {
        try {
            String result = aiService.detectHelmet(userId);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("An error occurred: " + e.getMessage());
        }
    }

    // 2인 이상 탑승 감지 요청 처리
    @PostMapping("/people-detection")
    public ResponseEntity<String> peopleDetection(@RequestParam String userId) {
        try {
            String result = aiService.detectPeople(userId);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("An error occurred: " + e.getMessage());
        }
    }

    // 횡단보도 감지 요청 처리
    @PostMapping("/crosswalk-detection")
    public ResponseEntity<String> crosswalkDetection(@RequestParam String userId) {
        try {
            String result = aiService.detectCrosswalk(userId);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("An error occurred: " + e.getMessage());
        }
    }
}
