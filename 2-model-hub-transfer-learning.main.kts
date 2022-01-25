#!/usr/bin/env kotlin

@file:DependsOn("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.3.0")

import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.dogsCatsSmallDatasetPath
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.sharpen
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformTensor
import java.io.File

val imageSize = 224L
val numberOfClasses = 10

val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
val mobileNetV2ModelType = TFModels.CV.MobileNetV2
val model = modelHub.loadModel(mobileNetV2ModelType)

val dogsVsCatsPath = dogsCatsSmallDatasetPath()

val preprocessing: Preprocessing = preprocess {
    load {
        pathToData = File(dogsVsCatsPath)
        imageShape = ImageShape(channels = 3L)
        colorMode = ColorOrder.BGR
        labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
    }
    transformImage {
        resize {
            outputHeight = imageSize.toInt()
            outputWidth = imageSize.toInt()
            interpolation = InterpolationType.BILINEAR
        }
    }
    transformTensor {
        sharpen {
            this.modelType = mobileNetV2ModelType
        }
    }
}

val dataset = OnHeapDataset.create(preprocessing).shuffle()
val (train, test) = dataset.split(.7)

model.use {
    it.layers.last().isTrainable = true

    it.compile(
        optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )
}

val layers = mutableListOf<Layer>()

model.layers.forEach { layer ->
    layer.isTrainable = false
    layers.add(layer)
}

val lastLayer = layers.last()
lastLayer.inboundLayers.forEach { inboundLayer ->
    inboundLayer.outboundLayers.remove(lastLayer)
}

layers.removeLast()

var x = Dense(name = "top_dense", outputSize = 200, activation = Activations.Selu)(lastLayer)
x = Dense(name = "pred", outputSize = 2, activation = Activations.Softmax)(x)

val modelWithCustomTop = Functional.fromOutput(x)

modelWithCustomTop.use {
    it.compile(
        optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    it.printSummary()

    val modelWeights = modelHub.loadWeights(mobileNetV2ModelType)
    it.loadWeightsForFrozenLayers(modelWeights)

    val accuracyBeforeTraining = it.evaluate(dataset = test, batchSize = 16).metrics[Metrics.ACCURACY]
    println("Accuracy before fine-tuning $accuracyBeforeTraining")

    it.fit(
        dataset = train,
        batchSize = 8,
        epochs = 3
    )

    val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 16).metrics[Metrics.ACCURACY]

    println("Accuracy after fine-tuning $accuracyAfterTraining")
}

