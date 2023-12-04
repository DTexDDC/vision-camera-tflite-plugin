import Foundation
import Vision
import TensorFlowLite
import Accelerate

@objc(TFLiteFrameProcessorPlugin)
public class TFLiteFrameProcessorPlugin: NSObject, FrameProcessorPluginBase {
    
  static let inputSize = 512
  static var isConfiguringModel = false
  static var configuredModelPath = ""
  static var modelInterpreter: Interpreter?
  
  static func logError(_ message: String) {
    #if DEBUG
    RCTDefaultLogFunction(.error, .native, #file, #line, "VisionCameraTFLitePlugin.\(#function): \(message)")
    #endif
  }
  
  static func configureModel(path: String) {
      isConfiguringModel = true
      if let uri = URL(string: path) {
        do {
          print("Loading model with path \(uri.path)")
          let interpreter = try Interpreter(modelPath: uri.path)
          try interpreter.allocateTensors()
          modelInterpreter = interpreter
        }
        catch {
          logError(error.localizedDescription)
        }
      }
      isConfiguringModel = false
    }
  
  static func processBoundingBoxes(input: [Float32]) -> [Dictionary<String, Float32>] {
      let flatBBs = input.chunked(by: 4)
      var out: [Dictionary<String, Float32>] = []
      for flatBB in flatBBs {
        out.append([
          "x": flatBB[1],
          "y": flatBB[0],
          "width": flatBB[3] - flatBB[1],
          "height": flatBB[2] - flatBB[0]
        ])
      }
      return out
    }
    
    @objc
    public static func callback(_ frame: Frame!, withArgs args: [Any]!) -> Any! {
      
      if isConfiguringModel == false {
        if let modelPath = args[0] as? String, modelPath != configuredModelPath {
          configureModel(path: modelPath)
          configuredModelPath = modelPath
        }
      }
      
      // Check there is a valid model, else return
      guard let model = modelInterpreter else {
        return nil
      }
      
      guard let imageBuffer = CMSampleBufferGetImageBuffer(frame.buffer) else {
        return nil
      }
      
      // TODO: Let vision camera use whatever pixel format and convert this on the
      // fly efficiently for the model input (right now it HAS to use 32BRGA)
      let sourcePixelFormat = CVPixelBufferGetPixelFormatType(imageBuffer)
      assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)
      
      
      do {
        let inputTensor = try model.input(at: 0)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputSize, height: inputSize)
        guard let scaledPixelBuffer = imageBuffer.resized(to: scaledSize) else {
          return nil
        }
        
        guard let inputData = rgbDataFromBuffer(
          scaledPixelBuffer,
          byteCount: 1 * inputSize * inputSize * 3,
          isModelQuantized: inputTensor.dataType == .uInt8
        ) else {
          print("Failed to convert the image buffer to RGB data.")
          return nil
        }
        
        // Copy the RGB data to the input `Tensor`.
        try model.copy(inputData, toInputAt: 0)
        try model.invoke()
        let confidenceOutput = try model.output(at: 0)
        let locationsOutput = try model.output(at: 1)
        let detectionCountOutput = try model.output(at: 2)
        let categoriesOutput = try model.output(at: 3)
        
        return [
          "scores": [Float32](unsafeData: confidenceOutput.data) ?? [],
          "boundingBoxes": processBoundingBoxes(input: [Float32](unsafeData: locationsOutput.data) ?? []),
          "detectionCount": [Float32](unsafeData: detectionCountOutput.data) ?? [],
          "categories": [Float32](unsafeData: categoriesOutput.data) ?? []
        ]
      }
      catch {
        return nil
      }
    }
    
}
