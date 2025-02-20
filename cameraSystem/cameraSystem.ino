#include "esp_camera.h"
#include <WiFi.h>
#include "wasteTemplates.h"
#include <TensorFlowLite_ESP32.h>
#include "waste_model.h" // 48000j model
#include "camera_pins.h"
#include "app_httpd.cpp"
#include "camera_index.h"
#include "camera_pins.h"

#define CAMERA_MODEL_AI_THINKER // Has PSRAM

const char *ssid = "PLDTHOMEFIBR6e080";
const char *password = "PLDTWIFIgz6q3";

void startCameraServer();
void setupLedFlash(int pin);

// Define WasteFeatures struct
struct WasteFeatures {
  String category;
  float confidence;
};

WasteFeatures classifyWasteUsingML(camera_fb_t *fb);
WasteTemplate findClosestTemplate(WasteFeatures features);

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing...");
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  
  startCameraServer();
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }
  
  WasteFeatures features = classifyWasteUsingML(fb);
  Serial.println("Detected Waste Category: " + features.category);
  Serial.println("Confidence: " + String(features.confidence));
  
  esp_camera_fb_return(fb);
  delay(8000);
}

WasteFeatures classifyWasteUsingML(camera_fb_t *fb) {
  WasteFeatures features;
  tflite::MicroInterpreter interpreter(waste_model, waste_tensor_arena, sizeof(waste_tensor_arena));
  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);
  memcpy(input->data.uint8, fb->buf, input->bytes);
  interpreter.Invoke();

  TfLiteTensor* output = interpreter.output(0);
  int bestIndex = 0;
  float bestConfidence = 0;
  for (int i = 0; i < output->dims->data[1]; i++) {
    float confidence = output->data.f[i];
    if (confidence > bestConfidence) {
      bestConfidence = confidence;
      bestIndex = i;
    }
  }

  String categories[] = {"Biodegradable", "Non-Biodegradable", "Recyclable"};
  features.category = categories[bestIndex];
  features.confidence = bestConfidence;
  return features;
}
