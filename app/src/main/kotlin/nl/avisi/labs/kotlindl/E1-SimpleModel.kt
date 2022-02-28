package nl.avisi.labs.kotlindl

import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.mnist
import kotlin.math.pow

object E1Constants {
    const val IMAGE_SIZE = 28L
    const val NUMBER_OF_CLASSES = 10
    const val BASE_CONV_FILTERS = 32

    const val BATCH_SIZE = 500
    const val EPOCHS = 10
}

fun main() {
    val model = createModel()

    val (train, validation, test) = getTrainValidationTestDatasets(validationSplit = 0.1)

    model.use {
        it.trainUsingDatasets(train, validation)
        it.evaluateUsingDataset(test)
    }
}

private fun createModel(): GraphTrainableModel {
    val layers = buildList {
        add(Input(dims = longArrayOf(E1Constants.IMAGE_SIZE, E1Constants.IMAGE_SIZE, 1)))

        repeat(2) { convolutionBlockNumber ->
            val numberOfFilters = E1Constants.BASE_CONV_FILTERS * 2.0.pow(convolutionBlockNumber).toLong()
            add(
                Conv2D(
                    filters = numberOfFilters,
                    kernelSize = longArrayOf(3, 3),
                    activation = Activations.Relu,
                    padding = ConvPadding.SAME,
                    kernelRegularizer = L2()
                )
            )
            add(AvgPool2D(poolSize = intArrayOf(1, 2, 2, 1)))
            add(Dropout(keepProbability = 0.7f))
        }

        add(Flatten())
        add(Dense(outputSize = E1Constants.NUMBER_OF_CLASSES, activation = Activations.Softmax))
    }
    return Sequential.of(layers)
}

private fun getTrainValidationTestDatasets(validationSplit: Double): Triple<OnHeapDataset, OnHeapDataset, OnHeapDataset> {
    val (trainVal, test) = mnist()

    val (train, validation) = trainVal.split(1.0 - validationSplit)
    return Triple(train, validation, test)
}

private fun GraphTrainableModel.trainUsingDatasets(trainData: OnHeapDataset, validationData: OnHeapDataset) {
    this.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY,
        callback = LoggingCallback()
    )

    this.printSummary()

    this.fit(
        trainingDataset = trainData,
        validationDataset = validationData,
        epochs = E1Constants.EPOCHS,
        trainBatchSize = E1Constants.BATCH_SIZE,
        validationBatchSize = E1Constants.BATCH_SIZE
    )
}

private fun GraphTrainableModel.evaluateUsingDataset(testData: OnHeapDataset) {
    val testResults = this.evaluate(dataset = testData, batchSize = E1Constants.BATCH_SIZE)

    val accuracy = testResults.metrics[Metrics.ACCURACY]
    val loss = testResults.lossValue

    println("Loss: $loss, accuracy: $accuracy")
}

class LoggingCallback : Callback() {
    override fun onEpochEnd(epoch: Int, event: EpochTrainingEvent, logs: TrainingHistory) {
        super.onEpochEnd(epoch, event, logs)

        println("Epoch $epoch, loss: ${event.lossValue}, acc: ${event.metricValue}, val_loss: ${event.valLossValue}, val_acc: ${event.valMetricValue}")
    }
}
