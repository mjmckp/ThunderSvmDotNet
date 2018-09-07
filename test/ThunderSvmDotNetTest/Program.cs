﻿using System;
using System.IO;
using ThunderSvmDotNet;

namespace ThunderSvmDotNetTest
{

    internal class Program
    {

        internal static double RandStdNormal(Random rng)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        internal static double RandNormal(Random rng, double mean, double stddev)
        {
            return mean + stddev * RandStdNormal(rng);
        }

        internal static (float[,], float []) GenerateData(Random rng, double radius, double sigma, int count, int numFeatures)
        {
            var data = new float[count, numFeatures];
            var labels = new float[count];

            for (var i = 0; i < count; i++)
            {
                var label = (rng.NextDouble() < 0.5);
                labels[i] = Convert.ToSingle(label);
                for (var j = 0; j < numFeatures; j++)
                {
                    if (label && j < 2)
                    {
                        var angle = 2 * Math.PI * rng.NextDouble();
                        data[i, j] = (float)(radius * ((j == 0) ? Math.Cos(angle) : Math.Sin(angle)) + RandNormal(rng, 0, sigma));
                    }
                    else
                        data[i, j] = (float)RandNormal(rng, 0, sigma);
                }
            }

            return (data, labels);
        }

        internal static int[,] ConfusionMatrix(float [] labels, float [] predictions)
        {
            var cm = new int[2, 2];
            for (var i = 0; i < labels.Length; i++)
            {
                var label = Convert.ToInt32(labels[i]);
                var pred = Convert.ToInt32(predictions[i]);
                cm[pred, label]++;
            }
            return cm;
        }

        private static void Main()
        {
            
            // Create random data
            var rng = new Random();
            const int trainCount = (int) 1e3;
            const int testCount = 100;
            const int numFeatures = 2;

            var radius = 1.0;
            var sigma = 0.1;

            var (trainData, trainLabels) = GenerateData(rng, radius, sigma, trainCount, numFeatures);
            var (testData, testLabels) = GenerateData(rng, radius, sigma, testCount, numFeatures);

            var file = Path.GetTempFileName();

            Console.WriteLine("Model training...");
            var prms = new Parameter(numFeatures) { Verbose = true };
            using (var model = Model.CreateDense(prms, trainData, trainLabels))
            {
                Console.WriteLine("Model trained.");
                if (model.NumClasses != 2) throw (new Exception("Expect 2 classes"));
                if (model.NumFeatures != numFeatures) throw (new Exception($"Expect {numFeatures} classes"));

                var predTrain = new float[trainCount];
                model.PredictDense(trainData, predTrain);
                var cmTrain = ConfusionMatrix(trainLabels, predTrain);

                var predTest = new float[testCount];
                model.PredictDense(testData, predTest);
                var cmTest = ConfusionMatrix(testLabels, predTest);

                Console.WriteLine($"Train: TN {cmTrain[0, 0]} TP {cmTrain[1, 1]} FN {cmTrain[0, 1]} FP {cmTrain[1, 0]}");
                Console.WriteLine($"Test:  TN {cmTest[0, 0]} TP {cmTest[1, 1]} FN {cmTest[0, 1]} FP {cmTest[1, 0]}");

                using (var stream = File.OpenWrite(file))
                {
                    using (var writer = new BinaryWriter(stream))
                    {
                        model.WriteBinary(writer);
                    }
                }
            }

            Model model2 = null;
            using (var stream = File.OpenRead(file))
            {
                using (var reader = new BinaryReader(stream))
                {
                    model2 = Model.ReadBinary(reader);
                }
            }
            using (model2)
            {
                if (model2.NumClasses != 2) throw (new Exception("Expect 2 classes"));
                if (model2.NumFeatures != numFeatures) throw (new Exception($"Expect {numFeatures} classes"));

                var predTrain = new float[trainCount];
                model2.PredictDense(trainData, predTrain);
                var cmTrain = ConfusionMatrix(trainLabels, predTrain);

                var predTest = new float[testCount];
                model2.PredictDense(testData, predTest);
                var cmTest = ConfusionMatrix(testLabels, predTest);

                Console.WriteLine($"Train2: TN {cmTrain[0, 0]} TP {cmTrain[1, 1]} FN {cmTrain[0, 1]} FP {cmTrain[1, 0]}");
                Console.WriteLine($"Test2:  TN {cmTest[0, 0]} TP {cmTest[1, 1]} FN {cmTest[0, 1]} FP {cmTest[1, 0]}");

            }

            Console.WriteLine("Done");
        }

    }

}