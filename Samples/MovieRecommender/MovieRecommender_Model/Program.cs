using Microsoft.ML;
using MovieRecommender_Model.Model;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

namespace MovieRecommender_Model;

/* This movie recommendation model is built on the http://files.grouplens.org/datasets/movielens/ml-latest-small.zip dataset
   for improved model performance use the https://grouplens.org/datasets/movielens/1m/ dataset instead. */

class Program
{
    private static readonly string BaseModelRelativePath = @"../../../Model";
    private static readonly string ModelRelativePath = $"{BaseModelRelativePath}/model.zip";

    private static readonly string BaseDataSetRelativepath = @"../../../Data";
    private static readonly string TrainingDataRelativePath = $"{BaseDataSetRelativepath}/ratings_train.csv";
    private static readonly string TestDataRelativePath = $"{BaseDataSetRelativepath}/ratings_test.csv";

    private static readonly string TrainingDataLocation = GetAbsolutePath(TrainingDataRelativePath);
    private static readonly string TestDataLocation = GetAbsolutePath(TestDataRelativePath);
    private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

    static void Main(string[] args)
    {

        //STEP 1: Create MLContext to be shared across the model creation workflow objects
        MLContext mlContext = new();

        //STEP 2: Read data from text file using TextLoader by defining the schema for reading the movie recommendation datasets and return dataview.
        var trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(path: TrainingDataLocation, hasHeader: true, separatorChar: ',');

        Console.WriteLine("=============== Reading Input Files ===============");
        Console.WriteLine();

        // ML.NET doesn't cache data set by default. Therefore, if one reads a data set from a file and accesses it many times, it can be slow due to
        // expensive featurization and disk operations. When the considered data can fit into memory, a solution is to cache the data in memory. Caching is especially
        // helpful when working with iterative algorithms which needs many data passes. Since SDCA is the case, we cache. Inserting a
        // cache step in a pipeline is also possible, please see the construction of pipeline below.
        trainingDataView = mlContext.Data.Cache(trainingDataView);

        Console.WriteLine("=============== Transform Data And Preview ===============");
        Console.WriteLine();

        //STEP 4: Transform your data by encoding the two features userId and movieID.
        //        These encoded features will be provided as input to FieldAwareFactorizationMachine learner
        var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "userIdFeaturized", inputColumnName: nameof(MovieRating.userId))
                                      .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "movieIdFeaturized", inputColumnName: nameof(MovieRating.movieId))
                                      .Append(mlContext.Transforms.Concatenate("Features", "userIdFeaturized", "movieIdFeaturized")));
        Common.ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 10);

        // STEP 5: Train the model fitting to the DataSet
        Console.WriteLine("=============== Training the model ===============");
        Console.WriteLine();
        var trainingPipeLine = dataProcessPipeline.Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(new string[] { "Features" }));
        var model = trainingPipeLine.Fit(trainingDataView);

        //STEP 6: Evaluate the model performance
        Console.WriteLine("=============== Evaluating the model ===============");
        Console.WriteLine();
        var testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(path: TestDataLocation, hasHeader: true, separatorChar: ',');

        var prediction = model.Transform(testDataView);

        var metrics = mlContext.BinaryClassification.Evaluate(data: prediction, labelColumnName: "Label", scoreColumnName: "Score", predictedLabelColumnName: "PredictedLabel");
        Console.WriteLine("Evaluation Metrics: acc:" + Math.Round(metrics.Accuracy, 2) + " AreaUnderRocCurve(AUC):" + Math.Round(metrics.AreaUnderRocCurve, 2));

        //STEP 7:  Try/test a single prediction by predicting a single movie rating for a specific user
        Console.WriteLine("=============== Test a single prediction ===============");
        Console.WriteLine();
        var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
        MovieRating testData = new MovieRating() { userId = "6", movieId = "10" };

        var movieRatingPrediction = predictionEngine.Predict(testData);
        Console.WriteLine($"UserId:{testData.userId} with movieId: {testData.movieId} Score:{Sigmoid(movieRatingPrediction.Score)} and Label {movieRatingPrediction.PredictedLabel}", Color.YellowGreen);
        Console.WriteLine();

        //STEP 8:  Save model to disk
        Console.WriteLine("=============== Writing model to the disk ===============");
        Console.WriteLine(); mlContext.Model.Save(model, trainingDataView.Schema, ModelPath);

        Console.WriteLine("=============== Re-Loading model from the disk ===============");
        Console.WriteLine();
        ITransformer trainedModel;
        using (FileStream stream = new(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
        }

        Console.WriteLine("Press any key to exit ...");
        Console.Read();
    }

    

    public static float Sigmoid(float x)
    {
        return (float)(100 / (1 + Math.Exp(-x)));
    }

    public static string GetAbsolutePath(string relativeDatasetPath)
    {
        FileInfo _dataRoot = new(typeof(Program).Assembly.Location);
        string assemblyFolderPath = _dataRoot.Directory.FullName;

        string fullPath = Path.Combine(assemblyFolderPath, relativeDatasetPath);

        return fullPath;
    }
}
