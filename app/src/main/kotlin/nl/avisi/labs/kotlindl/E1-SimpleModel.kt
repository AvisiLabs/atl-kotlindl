package nl.avisi.labs.kotlindl

import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
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
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset

object E1Constants {
    const val IMAGE_SIZE = 28L
    const val NUMBER_OF_CLASSES = 10
}

fun main() {
    val model = createModel()

    val (train, validation, test) = getTrainValidationTestDatasets(validationSplit = 0.1)

    model.use {
        it.trainUsingDatasets(train, validation)
        it.evaluateUsingDataset(test)
    }
}

private fun createModel() = Sequential.of(
    Input(dims = longArrayOf(E1Constants.IMAGE_SIZE, E1Constants.IMAGE_SIZE, 1)),
    Conv2D(filters = 6, kernelSize = longArrayOf(5, 5), activation = Activations.Selu),
    AvgPool2D(poolSize = intArrayOf(1, 4, 4, 1)),
    Flatten(),
    Dense(outputSize = 20, activation = Activations.Selu),
    Dense(outputSize = E1Constants.NUMBER_OF_CLASSES, activation = Activations.Softmax)
)

private fun getTrainValidationTestDatasets(validationSplit: Double): Triple<OnHeapDataset, OnHeapDataset, OnHeapDataset> {
    val (trainVal, test) = org.jetbrains.kotlinx.dl.dataset.mnist()

    val (train, validation) = trainVal.split(1.0 - validationSplit)
    return Triple(train, validation, test)
}

private fun GraphTrainableModel.trainUsingDatasets(trainData: OnHeapDataset, validationData: OnHeapDataset) {
    this.compile(
        optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    this.printSummary()

    this.fit(
        trainingDataset = trainData,
        validationDataset = validationData,
        epochs = 3,
        trainBatchSize = 500,
        validationBatchSize = 500
    )
}

private fun GraphTrainableModel.evaluateUsingDataset(testData: OnHeapDataset) {
    val testResults = this.evaluate(dataset = testData, batchSize = 500)

    val accuracy = testResults.metrics[Metrics.ACCURACY]
    val loss = testResults.lossValue

    println("Loss: $loss, accuracy: $accuracy")
}
