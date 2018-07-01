using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ClusteringSample
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string DataPath => @"C:\Experiment\mldotnet\data\iris-full.txt";//Path.Combine(AppPath, "datasets", "iris-full.txt");
        private static string ModelPath => Path.Combine(AppPath, "IrisClustersModel.zip");

        static List<IrisData> GetIrisDataSet()
        {
            var datas = new List<IrisData>();
            var lines = File.ReadAllLines(DataPath);
            int counter = 0;
            foreach(var line in lines)
            {
                counter++;
                if (counter > 1)
                {
                    var cols = Regex.Split(line, "	");
                    datas.Add(new IrisData() { PetalLength = Convert.ToSingle(cols[3]), PetalWidth = Convert.ToSingle(cols[4])
                        , SepalLength = Convert.ToSingle(cols[1]), SepalWidth = Convert.ToSingle(cols[2]) });
                }
            }
            return datas;
        }
        private static async Task Main(string[] args)
        {
            // STEP 1: Create a model
            var model = await TrainAsync();

            // STEP 2: Make a prediction
            Console.WriteLine();
            var prediction1 = model.Predict(TestIrisData.Setosa1);
            var prediction2 = model.Predict(TestIrisData.Setosa2);
            Console.WriteLine($"Clusters assigned for setosa flowers:");
            Console.WriteLine($"                                        {prediction1.SelectedClusterId}");
            Console.WriteLine($"                                        {prediction2.SelectedClusterId}");

            var prediction3 = model.Predict(TestIrisData.Virginica1);
            var prediction4 = model.Predict(TestIrisData.Virginica2);
            Console.WriteLine($"Clusters assigned for virginica flowers:");
            Console.WriteLine($"                                        {prediction3.SelectedClusterId}");
            Console.WriteLine($"                                        {prediction4.SelectedClusterId}");

            var prediction5 = model.Predict(TestIrisData.Versicolor1);
            var prediction6 = model.Predict(TestIrisData.Versicolor2);
            Console.WriteLine($"Clusters assigned for versicolor flowers:");
            Console.WriteLine($"                                        {prediction5.SelectedClusterId}");
            Console.WriteLine($"                                        {prediction6.SelectedClusterId}");
            Console.ReadLine();
        }

        internal static async Task<PredictionModel<IrisData, ClusterPrediction>> TrainAsync()
        {
            // LearningPipeline holds all steps of the learning process: data, transforms, learners.
            var pipeline = new LearningPipeline
                        {
                            // The TextLoader loads a dataset. The schema of the dataset is specified by passing a class containing
                            // all the column names and their types.
                            CollectionDataSource.Create<IrisData>(GetIrisDataSet()),
                            //new TextLoader(DataPath).CreateFrom<IrisData>(useHeader: true),
                            // ColumnConcatenator concatenates all columns into Features column
                            new ColumnConcatenator("Features",
                                "SepalLength",
                                "SepalWidth",
                                "PetalLength",
                                "PetalWidth"),
                            // KMeansPlusPlusClusterer is an algorithm that will be used to build clusters. We set the number of clusters to 3.
                            new KMeansPlusPlusClusterer() { K = 3 }
                        };

            Console.WriteLine("=============== Training model ===============");
            var model = pipeline.Train<IrisData, ClusterPrediction>();
            Console.WriteLine("=============== End training ===============");

            // Saving the model as a .zip file.
            await model.WriteAsync(ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);

            return model;
        }

    }


    internal class TestIrisData
    {
        internal static readonly IrisData Setosa1 = new IrisData()
        {
            SepalLength = 3.3f,
            SepalWidth = 1.6f,
            PetalLength = 0.2f,
            PetalWidth = 5.1f,
        };

        internal static readonly IrisData Setosa2 = new IrisData()
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f,
        };

        internal static readonly IrisData Virginica1 = new IrisData()
        {
            SepalLength = 3.1f,
            SepalWidth = 5.5f,
            PetalLength = 2.2f,
            PetalWidth = 6.4f,
        };

        internal static readonly IrisData Virginica2 = new IrisData()
        {
            SepalLength = 6.3f,
            SepalWidth = 3.3f,
            PetalLength = 6f,
            PetalWidth = 2.5f,
        };

        internal static readonly IrisData Versicolor1 = new IrisData()
        {
            SepalLength = 3.1f,
            SepalWidth = 4.5f,
            PetalLength = 1.5f,
            PetalWidth = 6.4f,
        };

        internal static readonly IrisData Versicolor2 = new IrisData()
        {
            SepalLength = 3.2f,
            SepalWidth = 4.7f,
            PetalLength = 1.4f,
            PetalWidth = 7.0f,
        };
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint SelectedClusterId;

        [ColumnName("Score")]
        public float[] Distance;
    }

    public class IrisData
    {
        [Column("0")]
        public float Label;

        [Column("1")]
        public float SepalLength;

        [Column("2")]
        public float SepalWidth;

        [Column("3")]
        public float PetalLength;

        [Column("4")]
        public float PetalWidth;
    }

}
