package com.example.safeT.kickboard;

import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.InputStreamReader;

@Service
public class AIService {

    // 파이썬 코드 실행 및 결과 반환
    private String executePythonScript(String scriptPath, String userId) {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("python", scriptPath, userId);
            Process process = processBuilder.start();
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder result = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line);
            }
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return result.toString();
            } else {
                return "Error executing Python script.";
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Exception occurred: " + e.getMessage();
        }
    }

    // 얼굴 인식 기능 호출
    public String sendUserIdToPython(String userId) {
        return executePythonScript("C:/safeT/ai_integration/Face_Recogniton/dataNface.py", userId);
    }

    // 헬멧 감지 기능 호출
    public String detectHelmet(String userId) {
        return executePythonScript("C:/safeT/ai_integration/Person_Detection/realtime_helmet_detection.py", userId);
    }

    // 2인 이상 탑승 감지 기능 호출
    public String detectPeople(String userId) {
        return executePythonScript("C:/safeT/ai_integration/Person_Detection/realtime_person_detection.py", userId);
    }

    // 횡단보도 감지 기능 호출
    public String detectCrosswalk(String userId) {
        return executePythonScript("C:/safeT/ai_integration/Crosswalk_Detection/crosswalk_detection.py", userId);
    }
}
