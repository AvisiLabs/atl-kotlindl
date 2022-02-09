package nl.avisi.labs.kotlindl

import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Input
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Output
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.SavedModel
import org.jetbrains.kotlinx.dl.dataset.mnist
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

fun main() {
    val (train, _) = mnist()
    val iiii = train.getX(0)
    val path = "/Users/marheerd/gitprojects/atl-kotlindl/custom_model"
    SavedModel.load(path).use {
        val image = it.loadImage("object-detection-image.jpg")
        it.reshape(image.width.toLong(), image.height.toLong(), 3L)
        it.input(Input.PLACEHOLDER)
        it.output(Output.ARGMAX)
        val array = FloatArray(image.width * image.height * 3)
        val floatArray = image.data.getPixels(0, 0, image.width, image.height, array)
        val prediction = it.predict(floatArray, "image_tensor", "detection_scores")
    }
}

private fun SavedModel.loadImage(image: String): BufferedImage {
    val imageURL = this::class.java.classLoader.getResource(image)
        ?: throw IllegalArgumentException("Could not get image $image from resources")

    return ImageIO.read(imageURL)
}
