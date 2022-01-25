#!/usr/bin/env kotlin

@file:DependsOn("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.3.0")

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.mnist

val imageSize = 28L
val numberOfClasses = 10

val model = Sequential.of(
    Input(dims = longArrayOf(imageSize, imageSize, 1)),
    Conv2D(filters = 6, kernelSize = longArrayOf(5, 5), activation = Activations.Selu),
    AvgPool2D(poolSize = intArrayOf(1, 4, 4, 1)),
    Flatten(),
    Dense(outputSize = 20, activation = Activations.Selu),
    Dense(outputSize = numberOfClasses, activation = Activations.Softmax)
)

val (trainVal, test) = mnist()

val (train, validation) = trainVal.split(.9)

model.use {
    it.compile(
        optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    it.printSummary()

    it.fit(
        trainingDataset = train,
        validationDataset = validation,
        epochs = 3,
        trainBatchSize = 500,
        validationBatchSize = 500
    )

    val testResults = it.evaluate(dataset = test, batchSize = 500)

    val accuracy = testResults.metrics[Metrics.ACCURACY]
    val loss = testResults.metrics[Metrics.ACCURACY]

    println("Loss: $loss, accuracy: $accuracy")
}
