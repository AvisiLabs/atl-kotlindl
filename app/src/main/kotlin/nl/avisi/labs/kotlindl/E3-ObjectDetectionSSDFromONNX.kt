package nl.avisi.labs.kotlindl

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Component
import java.awt.Dimension
import java.awt.Font
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.Stroke
import java.io.File
import javax.imageio.ImageIO
import javax.swing.JFrame

object E3Constants {
    const val TOP_K = 5
    const val EXAMPLE_IMAGE_RESOURCE_PATH = "object-detection-image.jpg"
}

fun main() {
    val modelType = ONNXModels.ObjectDetection.SSD
    val model = getModelFromModelHub(modelType) as SSDObjectDetectionModel

    model.use {
        it.detectObjectsForImage(E3Constants.EXAMPLE_IMAGE_RESOURCE_PATH)
    }
}

private fun getModelFromModelHub(modelType: ModelType<*, *>): InferenceModel {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    return modelHub.loadPretrainedModel(modelType)
}

private fun SSDObjectDetectionModel.detectObjectsForImage(image: String) {
    val imageURL = this::class.java.classLoader.getResource(image)
        ?: throw IllegalArgumentException("Could not get image $image from resources")

    val imageFile = File(imageURL.toURI())
    val detectedObjects = this.detectObjects(imageFile = imageFile, topK = E3Constants.TOP_K)

    detectedObjects.forEach {
        println("${it.classLabel} found at ${it.xMin} ${it.yMin} ${it.xMax} ${it.yMax} with probability ${it.probability}")
    }

    visualise(imageFile, detectedObjects)
}


private fun visualise(
    imageFile: File,
    detectedObjects: List<DetectedObject>
) {
    val frame = JFrame("Detected Objects")
    frame.contentPane.add(JPanel(imageFile, detectedObjects))
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}

class JPanel(
    image: File,
    private val detectedObjects: List<DetectedObject>
) : Component() {
    companion object {
        private const val WIDTH = 800
        private const val HEIGHT = 600
    }

    private val bufferedImage = ImageIO.read(image)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)

        graphics.drawImage(bufferedImage, 0, 0, WIDTH, HEIGHT, null)

        detectedObjects.forEach {
            val top = it.yMin * HEIGHT
            val left = it.xMin * WIDTH
            val bottom = it.yMax * HEIGHT
            val right = it.xMax * WIDTH

            graphics.color = Color.ORANGE
            graphics.font = Font("Courier New", 1, 17)
            graphics.drawString(" ${it.classLabel} : ${it.probability}", left.toInt(), bottom.toInt() - 8)

            graphics as Graphics2D
            val stroke: Stroke = BasicStroke(6f)
            graphics.color = when (it.classLabel) {
                "person" -> Color.RED
                "car" -> Color.GREEN
                "bicycle" -> Color.MAGENTA
                else -> Color.WHITE
            }
            graphics.stroke = stroke
            graphics.drawRect(left.toInt(), bottom.toInt(), (right - left).toInt(), (top - bottom).toInt())
        }
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(WIDTH, HEIGHT)
    }

    override fun getMinimumSize(): Dimension {
        return Dimension(WIDTH, HEIGHT)
    }
}
