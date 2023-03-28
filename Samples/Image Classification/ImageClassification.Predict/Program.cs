using System;
using System.IO;
using System.Linq;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.Predict
{
    internal class Program
    {
        private static void Main()
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
            var assetsPath = GetAbsolutePath(assetsRelativePath);

            var imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "images-for-predictions");

            var imageClassifierModelZipFilePath = Path.Combine(assetsPath, "inputs", "MLNETModel", "imageClassifier.zip");

            try
            {
                var mlContext = new MLContext(seed: 1);

                Console.WriteLine($"Loading model from: {imageClassifierModelZipFilePath}");

                // Load the model
                var loadedModel = mlContext.Model.Load(imageClassifierModelZipFilePath, out var modelInputSchema);

                // Create prediction engine to try a single prediction (input = ImageData, output = ImagePrediction)
                var predictionEngine = mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(loadedModel);

                //Predict the first image in the folder
                var imagesToPredict = FileUtils.LoadInMemoryImagesFromDirectory(imagesFolderPathForPredictions, false);


                Console.WriteLine("Predicting several images...");

                foreach (var currentImageToPredict in imagesToPredict)
                {
                    var currentPrediction = predictionEngine.Predict(currentImageToPredict);

                    Console.WriteLine(
                        $"Image Filename : [{currentImageToPredict.ImageFileName}], " +
                        $"Predicted Label : [{currentPrediction.PredictedLabel}], " +
                        $"Probability : [{currentPrediction.Score.Max()}]");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);
    }
}
