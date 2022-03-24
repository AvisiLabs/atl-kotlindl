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
    JFrame("Detected objects").apply {
        contentPane.add(JPanel(imageFile, detectedObjects))
        pack()
        setLocationRelativeTo(null)

        isVisible = true
        isResizable = true
        defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    }
}

class JPanel(
    image: File,
    private val detectedObjects: List<DetectedObject>
) : Component() {
    companion object {
        private const val WIDTH = 800
        private const val HEIGHT = 600
        private const val LABEL_FONT = "Courier New"
        private const val LABEL_SIZE = 17
        private const val RECTANGLE_STROKE_WIDTH = 6f
    }

    private val bufferedImage = ImageIO.read(image)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)

        graphics.drawImage(bufferedImage, 0, 0, WIDTH, HEIGHT, null)

        detectedObjects.forEach {
            drawObject(graphics as Graphics2D, it)
        }
    }

    private fun drawObject(graphics: Graphics2D, detectedObject: DetectedObject) {
        val top = detectedObject.yMin * HEIGHT
        val left = detectedObject.xMin * WIDTH
        val bottom = detectedObject.yMax * HEIGHT
        val right = detectedObject.xMax * WIDTH

        drawLabelForObject(graphics, detectedObject, left, bottom)

        drawRectangleForObject(graphics, detectedObject, left, bottom, right, top)
    }

    private fun drawLabelForObject(
        graphics: Graphics,
        detectedObject: DetectedObject,
        left: Float,
        bottom: Float
    ) {
        graphics.color = Color.ORANGE
        graphics.font = Font(LABEL_FONT, 1, LABEL_SIZE)
        graphics.drawString(
            " ${detectedObject.classLabel} : ${detectedObject.probability}",
            left.toInt(),
            bottom.toInt() - (LABEL_SIZE / 2)
        )
    }

    private fun drawRectangleForObject(
        graphics: Graphics2D,
        detectedObject: DetectedObject,
        left: Float,
        bottom: Float,
        right: Float,
        top: Float
    ) {
        val stroke: Stroke = BasicStroke(RECTANGLE_STROKE_WIDTH)
        graphics.color = determineRectangleColorForObject(detectedObject)
        graphics.stroke = stroke
        graphics.drawRect(left.toInt(), bottom.toInt(), (right - left).toInt(), (top - bottom).toInt())
    }

    private fun determineRectangleColorForObject(detectedObject: DetectedObject) = when (detectedObject.classLabel) {
        "person" -> Color.RED
        "car" -> Color.GREEN
        "bicycle" -> Color.MAGENTA
        else -> Color.WHITE
    }

    override fun getPreferredSize(): Dimension = Dimension(WIDTH, HEIGHT)
    override fun getMinimumSize(): Dimension = Dimension(WIDTH, HEIGHT)
}
