package nl.avisi.labs.kotlindl

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import java.io.File

object E5Constants {
    const val MODEL_CONFIG_PATH = "./models/custom_model_h5/model.json"
    const val MODEL_PATH = "./models/custom_model_h5/model_checkpoint.h5"
    const val IMAGE_RESOURCE_PATH = "beer.jpg"
}

/*
Model that works perfectly fine with Keras does not load in KotlinDL due to
failing layer pattern check in TensorFlow. Even after replacing forward slashes
it still fails due to missing inbound nodes.

Model (loaded directly from h5) works fine in Python with Keras
 */
fun main() {
    val image = ImageConverter.toNormalizedFloatArray(Sequential::class.java.classLoader.getResourceAsStream(E5Constants.IMAGE_RESOURCE_PATH)!!)
    Functional.loadModelConfiguration(File(E5Constants.MODEL_CONFIG_PATH)).use {
        it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
        it.loadWeights(HdfFile(File(E5Constants.MODEL_PATH)))
        val predictions = it.predict(image)
    }
}
