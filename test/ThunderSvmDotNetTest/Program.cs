using System;
using System.IO;
using System.Collections.Generic;
using ThunderSvmDotNet;

namespace ThunderSvmDotNetTest
{

    internal class Program
    {
        /// <summary>
        /// Returns the row with number 'row' of this matrix as a 1D-Array.
        /// </summary>
        internal static T[] GetRow<T>(T[,] matrix, int row)
        {
            var rowLength = matrix.GetLength(1);
            var rowVector = new T[rowLength];

            for (var i = 0; i < rowLength; i++)
                rowVector[i] = matrix[row, i];

            return rowVector;
        }

        internal static SparseMatrix Dense2Sparse(float[,] matrix)
        {
            var nnz = 0;
            for (var i = 0; i < matrix.GetLength(0); i++)
                for (var j = 0; j < matrix.GetLength(1); j++)
                    if (matrix[i, j] != 0) nnz++;

            var rslt = new SparseMatrix(matrix.GetLength(0), matrix.GetLength(1), nnz);

            var idx = 0;
            for (var i = 0; i < matrix.GetLength(0); i++)
            {
                rslt.RowExtents[i] = idx;
                for (var j = 0; j < matrix.GetLength(1); j++)
                {
                    if (matrix[i, j] != 0)
                    {
                        rslt.Data[idx] = matrix[i, j];
                        rslt.ColumnIndices[idx] = j;
                        idx++;
                    }
                }
            }
            rslt.RowExtents[rslt.RowExtents.Length-1] = idx;

            rslt.Validate();

            return rslt;
        }

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
                    else if (rng.Next(20) > 0)  // insert some zeros for sparse test
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
            try
            {
                Console.WriteLine("Dense Model training...");
                var prms = new Parameter(numFeatures) { Verbose = true };
                using (var model = Model.CreateDense(prms, trainData, trainLabels))
                {
                    Console.WriteLine("Dense Model trained.");
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

                    model.Parameter.Verbose = false;
                    for (var i = 0; i < Math.Min(100,trainCount); i++)
                    {
                        var row = GetRow(trainData, i);
                        var pi = model.PredictDense(row);
                        if (pi != predTrain[i])
                            throw (new Exception($"Predict mismatch: {pi} != {predTrain[i]}"));
                    }

                    using (var stream = File.OpenWrite(file))
                    {
                        using (var writer = new BinaryWriter(stream))
                        {
                            model.WriteBinary(writer);
                        }
                    }
                }

                // Check that we can persist model to file and back again
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

                    File.Delete(file);
                    model2.Export(file);
                }

                // Check that we can export model to file and then load it into a LibSvmDotNet model, and get the same results
                using (var model3 = LibSvmDotNet.Model.Load(file))
                {
                    var trainData3 = new List<LibSvmDotNet.Node[]>(trainCount);
                    for (var i = 0; i < trainCount; i++)
                    {
                        var row = new LibSvmDotNet.Node[numFeatures];
                        for (var j = 0; j < numFeatures; j++)
                            row[j] = new LibSvmDotNet.Node() { Index = j, Value = trainData[i, j] };
                        trainData3.Add(row);
                    }

                    var predTrain3 = new float[trainCount];
                    using (var problem = LibSvmDotNet.Problem.FromSequence(trainData3, new double[trainCount]))
                    {
                        for(var i=0; i<problem.Length; i++)
                        {
                            predTrain3[i] = (float)LibSvmDotNet.LibSvm.Predict(model3, problem.X[i]);
                        }
                    }

                    var cmTrain3 = ConfusionMatrix(trainLabels, predTrain3);
                    Console.WriteLine($"Train3: TN {cmTrain3[0, 0]} TP {cmTrain3[1, 1]} FN {cmTrain3[0, 1]} FP {cmTrain3[1, 0]}");
                }

                // Compare with sparse version
                Console.WriteLine("Sparse Model training...");
                var trainDataSparse = Dense2Sparse(trainData);
                var testDataSparse = Dense2Sparse(testData);
                using (var model = Model.CreateSparse(prms, trainDataSparse, trainLabels))
                {
                    Console.WriteLine("Sparse Model trained.");
                    if (model.NumClasses != 2) throw (new Exception("Expect 2 classes"));
                    if (model.NumFeatures != numFeatures) throw (new Exception($"Expect {numFeatures} classes"));

                    var predTrain = new float[trainCount];
                    model.PredictSparse(trainDataSparse, predTrain);
                    var cmTrain = ConfusionMatrix(trainLabels, predTrain);

                    var predTest = new float[testCount];
                    model.PredictSparse(testDataSparse, predTest);
                    var cmTest = ConfusionMatrix(testLabels, predTest);

                    Console.WriteLine($"Train: TN {cmTrain[0, 0]} TP {cmTrain[1, 1]} FN {cmTrain[0, 1]} FP {cmTrain[1, 0]}");
                    Console.WriteLine($"Test:  TN {cmTest[0, 0]} TP {cmTest[1, 1]} FN {cmTest[0, 1]} FP {cmTest[1, 0]}");
                }

                Console.WriteLine("Done");
            }
            finally
            {
                if (File.Exists(file))
                    File.Delete(file);
            }
        }

    }

}
