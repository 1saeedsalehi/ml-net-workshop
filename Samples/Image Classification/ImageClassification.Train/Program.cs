using ImageClassification.DataModels;
using ImageClassification.Train.DataModels;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ImageClassification.Train
{
    internal class Program
    {
        static void Main()
        {
            //just for looking geek 😎
            Console.WriteLine(@"

███╗░░░███╗██╗███╗░░██╗██████╗░░██████╗██╗░░██╗░█████╗░██████╗░██████╗░
████╗░████║██║████╗░██║██╔══██╗██╔════╝██║░░██║██╔══██╗██╔══██╗██╔══██╗
██╔████╔██║██║██╔██╗██║██║░░██║╚█████╗░███████║███████║██████╔╝██████╔╝
██║╚██╔╝██║██║██║╚████║██║░░██║░╚═══██╗██╔══██║██╔══██║██╔══██╗██╔═══╝░
██║░╚═╝░██║██║██║░╚███║██████╔╝██████╔╝██║░░██║██║░░██║██║░░██║██║░░░░░
╚═╝░░░░░╚═╝╚═╝╚═╝░░╚══╝╚═════╝░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░
");
            const string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");
            string imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "test-images");

            string imageSetFolderPath = Path.Combine(assetsPath, "inputs", "images", "flower_photos_small_set");


            var mlContext = new MLContext(seed: 1);
            mlContext.Log += MLContextLog;

            // 1. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: imageSetFolderPath, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 2. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion.
                    MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: imageSetFolderPath,
                                                inputColumnName: "ImagePath"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            // 3. Split the data 80:20 into train and test sets, train and evaluate.
            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            // 5. Define the model's training pipeline using DNN default values
            var pipeline = mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));

            // 4.  Train/create the ML model
            //in order to use TensorFlow you need to install Microsoft Visual C++ Redistributable
            //download link: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170

            ITransformer trainedModel = pipeline.Fit(trainDataView);


            // 5. Get the quality metrics (accuracy, etc.)
            EvaluateModel(mlContext, testDataView, trainedModel);

            // 6. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Model saved to: {outputMlNetModelFilePath}");

            // 7. Try a single prediction simulating an end-user app
            TrySinglePrediction(imagesFolderPathForPredictions, mlContext, trainedModel);

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for TensorFlow DNN Transfer Learning multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");

            int i = 0;
            foreach (var classLogLoss in metrics.PerClassLogLoss)
            {
                i++;
                Console.WriteLine($"    LogLoss for class {i} = {classLogLoss:0.####}, the closer to 0, the better");
            }
            Console.WriteLine($"************************************************************");


        }

        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            var testImages = FileUtils.LoadInMemoryImagesFromDirectory(
                imagesFolderPathForPredictions, false);

            var imageToPredict = testImages.First();

            var prediction = predictionEngine.Predict(imageToPredict);

            Console.WriteLine(
                $"Image Filename : [{imageToPredict.ImageFileName}], " +
                $"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");
        }


        public static IEnumerable<ImageData> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData(x.imagePath, x.label));

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);



        private static void MLContextLog(object sender, LoggingEventArgs e)
        {
            Console.WriteLine(e.Message);
        }
    }
}

