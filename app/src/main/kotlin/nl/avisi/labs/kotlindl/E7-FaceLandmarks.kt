package nl.avisi.labs.kotlindl


import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.rescale
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformTensor
import org.jetbrains.kotlinx.dl.visualization.swing.drawLandMarks
import java.io.File

object E7Constants {
    const val IMAGE_RESOURCE_PATH = "face-landmarks.png"
    const val IMAGE_SIZE = 448L
}

fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.FaceAlignment.Fan2d106.pretrainedModel(modelHub)

    val imageUri = ONNXModelHub::class.java.classLoader.getResource(E7Constants.IMAGE_RESOURCE_PATH)!!.toURI()
    val imageFile = File(imageUri)

    model.use {
        val landmarks = it.detectLandmarks(imageFile = imageFile)
        visualiseLandMarks(imageFile, landmarks)
    }
}

fun visualiseLandMarks(
    imageFile: File,
    landmarks: List<Landmark>
) {
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageFile
            imageShape = ImageShape(E7Constants.IMAGE_SIZE, E7Constants.IMAGE_SIZE, 3)
            colorMode = ColorOrder.BGR
        }
        transformImage {
            resize {
                outputWidth = E7Constants.IMAGE_SIZE.toInt()
                outputHeight = E7Constants.IMAGE_SIZE.toInt()
            }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val rawImage = preprocessing().first

    drawLandMarks(rawImage, ImageShape(E7Constants.IMAGE_SIZE, E7Constants.IMAGE_SIZE, 3), landmarks)
}
