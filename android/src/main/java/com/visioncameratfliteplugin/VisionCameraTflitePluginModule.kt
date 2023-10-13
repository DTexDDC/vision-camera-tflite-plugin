@file:Suppress("DEPRECATION")

package com.visioncameratfliteplugin

import android.annotation.SuppressLint
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.camera.core.ImageProxy
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.facebook.react.bridge.WritableNativeArray
import com.facebook.react.bridge.WritableNativeMap
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin
import com.visioncameratfliteplugin.utils.YuvToRgbConverter
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.File
import java.io.FileInputStream
import java.lang.Float
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteFrameProcessorPlugin(reactContext: ReactApplicationContext) : FrameProcessorPlugin("detectLabel") {

    private var isConfiguringModel = false
    private lateinit var interpreter: Interpreter
    private val inputSize = 512
    private val yuvToRgbConverter = YuvToRgbConverter(reactContext)
    private val detectionsSize = 25

    private lateinit var bitmap: Bitmap

    /** Memory-map the model file in Assets.  */
    private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun processBoundingBoxes(input: Array<FloatArray>): WritableArray {
        val out = Arguments.createArray()
        for (bb in input) {
            val map = Arguments.createMap()
            map.putDouble("x", bb[1].toDouble())
            map.putDouble("y", bb[0].toDouble())
            map.putDouble("width", bb[3].toDouble() - bb[1].toDouble())
            map.putDouble("height", bb[2].toDouble() - bb[0].toDouble())
            out.pushMap(map)
        }
        return out
    }

    private fun configureModel(modelPath: String) {
        isConfiguringModel = true
        try {
            val uri = Uri.parse(modelPath)
            if (uri.scheme != "file") throw Error("Model path must be a file URI!")
            interpreter = Interpreter(File(uri.path))
            interpreter.allocateTensors()
        } catch (e: Exception) {
            throw RuntimeException(e);
        }
        isConfiguringModel = false
    }

    @SuppressLint("UnsafeOptInUsageError")
    override fun callback(image: ImageProxy, params: Array<Any>): Any? {
        if (!::interpreter.isInitialized && !isConfiguringModel) {
            val modelPath = params[0] as? String ?: return null
            configureModel(modelPath)
            return null
        }
        try {
            val result = WritableNativeMap()
            // Only init the bitmap once so we don't re-allocate on each frame
            if (!::bitmap.isInitialized) {
                bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                return null
            }
            // TFLite only takes RGB and CameraX uses YUV so convert
            yuvToRgbConverter.yuvToRgb(image.image!!, bitmap)
            // Preprocess image to input tensor size
            val imageRotation = image.imageInfo.rotationDegrees
            val imageProcessor = ImageProcessor.Builder()
                    .add(Rot90Op(-imageRotation / 90))
                    .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                    .build()
            // Create inputs and outputs and invoke model
            val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))
            val inputs = arrayOf(tensorImage.buffer)
            val scores = arrayOf(FloatArray(detectionsSize))
            val boundingBoxes = arrayOf(Array(detectionsSize) { FloatArray(4) })
            val detectionCount = FloatArray(1)
            val categories = arrayOf(FloatArray(detectionsSize))
            val outputs = mapOf(
                    0 to scores,
                    1 to boundingBoxes,
                    2 to detectionCount,
                    3 to categories
            )
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
            // Return outputs
            result.putArray("scores", Arguments.fromArray(scores[0]))
            result.putArray("boundingBoxes", processBoundingBoxes(boundingBoxes[0]))
            result.putInt("detectionCount", detectionCount[0].toInt())
            result.putArray("categories", Arguments.fromArray(categories[0]))
            return result
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
}
