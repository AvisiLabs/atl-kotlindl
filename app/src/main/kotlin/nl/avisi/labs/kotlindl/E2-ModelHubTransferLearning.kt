package nl.avisi.labs.kotlindl

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
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

object E2Constants {
    const val IMAGE_SIZE = 224L
    const val BATCH_SIZE = 32
}

fun main() {
    val modelType = TFModels.CV.ResNet18

    val (model, weights) = getModelAndWeightsFromHub(modelType)
    val modelWithCustomTop = createModelWithCustomClassifier(model)

    val dogsVsCatsPath = dogsCatsSmallDatasetPath()
    val preprocessing = createPreprocessingPipeline(dogsVsCatsPath, modelType)
    val (train, validation, test) = preprocessing.getTrainValidationTestDatasets(validationSplit = 0.2, testSplit = 0.2)

    modelWithCustomTop.use {
        it.trainAndValidateUsingDatasets(weights, train, validation, test)
    }
}

private fun createPreprocessingPipeline(datasetPath: String, modelType: ModelType<*, *>): Preprocessing =
    preprocess {
        load {
            pathToData = File(datasetPath)
            imageShape = ImageShape(channels = 3L)
            colorMode = ColorOrder.BGR
            labelGenerator = FromFolders(mapping = mapOf("cat" to 0, "dog" to 1))
        }
        transformImage {
            resize {
                outputHeight = E2Constants.IMAGE_SIZE.toInt()
                outputWidth = E2Constants.IMAGE_SIZE.toInt()
                interpolation = InterpolationType.BILINEAR
            }
        }
        transformTensor {
            sharpen {
                this.modelType = modelType
            }
        }
    }

private fun getModelAndWeightsFromHub(modelType: ModelType<out GraphTrainableModel, *>): Pair<GraphTrainableModel, HdfFile> {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val modelWeights = modelHub.loadWeights(modelType)
    return model to modelWeights
}

private fun createModelWithCustomClassifier(model: GraphTrainableModel): GraphTrainableModel {
    val layers = buildList {
        model.layers.forEach { layer ->
            if (layer.name == "fc1") {
                layer.inboundLayers.forEach { inboundLayer ->
                    inboundLayer.outboundLayers.remove(layer)
                }
                return@buildList
            }
            layer.isTrainable = false
            add(layer)
        }
    }

    var x = Dense(name = "top_dense", outputSize = 30, activation = Activations.Selu)(layers.last())
    x = Dense(name = "top_dense_2", outputSize = 20, activation = Activations.Selu)(x)
    x = Dense(name = "pred", outputSize = 2, activation = Activations.Softmax)(x)

    return Functional.fromOutput(x)
}

private fun Preprocessing.getTrainValidationTestDatasets(validationSplit: Double, testSplit: Double): Triple<OnHeapDataset, OnHeapDataset, OnHeapDataset> {
    val dataset = OnHeapDataset.create(this).shuffle()
    val (trainVal, test) = dataset.split(1.0 - testSplit)
    val (train, validation) = trainVal.split(1.0 - validationSplit)
    return Triple(train, validation, test)
}

private fun GraphTrainableModel.trainAndValidateUsingDatasets(
    weights: HdfFile,
    trainData: OnHeapDataset,
    validationData: OnHeapDataset,
    testData: OnHeapDataset,
) {
    this.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )

    this.printSummary()

    this.loadWeightsForFrozenLayers(weights)

    val accuracyBeforeTraining = this.evaluate(dataset = testData, batchSize = E2Constants.BATCH_SIZE).metrics[Metrics.ACCURACY]
    println("Accuracy before fine-tuning $accuracyBeforeTraining")

    this.fit(
        trainingDataset = trainData,
        validationDataset = validationData,
        trainBatchSize = E2Constants.BATCH_SIZE,
        validationBatchSize = E2Constants.BATCH_SIZE,
        epochs = 25
    )

    val accuracyAfterTraining = this.evaluate(dataset = testData, batchSize = E2Constants.BATCH_SIZE).metrics[Metrics.ACCURACY]

    println("Accuracy after fine-tuning $accuracyAfterTraining")
}
