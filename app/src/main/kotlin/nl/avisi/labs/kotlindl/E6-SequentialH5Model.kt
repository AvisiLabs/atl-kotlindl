package nl.avisi.labs.kotlindl

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import java.awt.image.DataBufferByte
import java.io.File

object E6Constants {
    const val MODEL_CONFIG_PATH = "./models/custom_model_h5_simple/mnist.json"
    const val MODEL_WEIGHTS_PATH = "./models/custom_model_h5_simple/mnist.h5"
    const val EXAMPLE_IMAGE_RESOURCE_PATH = "mnist_3.jpg"
}

/*
Model made in Keras:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
 */
fun main() {
    val image = ImageConverter.toBufferedImage(Sequential::class.java.classLoader.getResourceAsStream(E6Constants.EXAMPLE_IMAGE_RESOURCE_PATH)!!)
    val imageArray = OnHeapDataset.toNormalizedVector((image.raster.dataBuffer as DataBufferByte).data)
    Sequential.loadModelConfiguration(File(E6Constants.MODEL_CONFIG_PATH)).use {
        it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
        it.loadWeights(HdfFile(File(E6Constants.MODEL_WEIGHTS_PATH)))
        val prediction = it.predict(imageArray)
        println(prediction)
    }
}
