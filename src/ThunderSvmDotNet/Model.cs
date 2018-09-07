using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ThunderSvmDotNet.Interop;

// To build native binaries:
//    cmake .. -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE -G "Visual Studio 15 2017 Win64" -Tv140,host=x64
//    msbuild /verbosity:quiet /p:Configuration=Debug;Platform="x64" /m thundersvm.sln

namespace ThunderSvmDotNet
{
    /// <summary>
    /// Represents a trained model.
    /// </summary>
    public sealed class Model : DisposableObject
    {
        #region Constructors

        /// <summary>
        /// Create a new model of the given type
        /// </summary>
        internal unsafe Model(SvmType svmType)
        {
            this.NativePtr = NativeMethods.model_new((int)svmType);
        }

        internal unsafe Model(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new ArgumentException($"{nameof(ptr)} should not be IntPtr.Zero");
            this.NativePtr = ptr;

        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the number of classes.
        /// </summary>
        public int NumClasses
        {
            get; internal set;
        }

        /// <summary>
        /// Gets the number of features.
        /// </summary>
        public int NumFeatures
        {
            get; internal set;
        }

        /// <summary>
        /// Gets the parameter of this model.
        /// </summary>
        public Parameter Parameter
        {
            get; internal set;
        }

        /// <summary>
        /// Get the number of support vectors in the trained model.
        /// </summary>
        public int NumSupportVectors 
        {
            get
            {
                if (NativePtr == IntPtr.Zero)
                    throw (new Exception("Model not initialised"));
                ThrowIfDisposed();
                return NativeMethods.n_sv(NativePtr);
            }
        }

        /// <summary>
        /// Get the number of underlying binary models in the trained model.
        /// </summary>
        public unsafe int NumBinaryModels
        {
            get
            {
                if (NativePtr == IntPtr.Zero)
                    throw (new Exception("Model not initialised"));
                ThrowIfDisposed();
                var n_model = 0;
                NativeMethods.get_n_binary_models(NativePtr, &n_model);
                return n_model;
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Train an SVM model on a dense dataset.
        /// </summary>
        /// <param name="prms">Calibration parameters</param>
        /// <param name="data">Feature vectors (stack as rows)</param>
        /// <param name="label"></param>
        /// <returns></returns>
        public static unsafe Model CreateDense(Parameter prms, float [,] data, float [] label)
        {
            if (data == null)
                throw (new ArgumentException("Data matrix is null"));

            if (label != null && data.GetLength(0) != label.Length)
                throw (new ArgumentException("Number of rows in data matrix must match number of labels"));

            if (prms.LengthOfWeight > 0)
            {
                if (prms.Weight == null || prms.Weight.Length != prms.LengthOfWeight)
                    throw (new ArgumentException("LengthOfWeight does not match length of weight array"));

                if (prms.WeightLabel == null || prms.WeightLabel.Length != prms.LengthOfWeight)
                    throw (new ArgumentException("LengthOfWeight does not match length of weight label array"));
            }

            var gcHandles = new List<GCHandle>(4);

            var model = new Model(prms.SvmType);
            try
            {
                var dataHdl = GCHandle.Alloc(data, GCHandleType.Pinned);
                gcHandles.Add(dataHdl);
                var dataPtr = (float*)dataHdl.AddrOfPinnedObject().ToPointer();

                float *labelPtr = null;
                if (label != null)
                {
                    var labelHdl = GCHandle.Alloc(label, GCHandleType.Pinned);
                    gcHandles.Add(labelHdl);
                    labelPtr = (float*)labelHdl.AddrOfPinnedObject().ToPointer();
                }

                int* weightLabelPtr = null;
                float* weightPtr = null;
                if (prms.LengthOfWeight > 0)
                {
                    var weightLabelHdl = GCHandle.Alloc(prms.WeightLabel, GCHandleType.Pinned);
                    gcHandles.Add(weightLabelHdl);
                    weightLabelPtr = (int*)weightLabelHdl.AddrOfPinnedObject().ToPointer();

                    var weightHdl = GCHandle.Alloc(prms.Weight, GCHandleType.Pinned);
                    gcHandles.Add(weightHdl);
                    weightPtr = (float*)weightHdl.AddrOfPinnedObject().ToPointer();
                }

                int n_features = 0;
                int n_classes = 0;
                int succeed = 0;
                NativeMethods.dense_model_scikit(data.GetLength(0),            // int row_size, 
                                                 data.GetLength(1),            // int features,
                                                 dataPtr,                      // float* data,   // length: row_size * features (row-wise)
                                                 labelPtr,                     // float* label,  // length: row_size (can also be null)
                                                 (int)prms.SvmType,            // int svm_type,
                                                 (int)prms.KernelType,         // int kernel_type,
                                                 prms.Degree,                  // int degree,
                                                 (float)prms.Gamma,            // float gamma,
                                                 (float)prms.Coef0,            // float coef0,
                                                 (float)prms.C,                // float cost,       // param_cmd.C = (float_type)cost;
                                                 (float)prms.Nu,               // float nu,
                                                 (float)prms.P,                // float epsilon,    // param_cmd.p = (float_type)epsilon;
                                                 (float)prms.Epsilon,          // float tol,        // param_cmd.epsilon = (float_type)tol;
                                                 (prms.Probability ? 1 : 0),   // int probability,
                                                 prms.LengthOfWeight,          // int weight_size,
                                                 weightLabelPtr,               // int* weight_label, // length: weight_size
                                                 weightPtr,                    // float* weight,     // length: weight_size
                                                 (prms.Verbose ? 1 : 0),       // int verbose,
                                                 prms.MaxIter,                 // int max_iter,
                                                 prms.NumCores,                // int n_cores,
                                                 prms.MaxMemSize,              // int max_mem_size,
                                                 &n_features,                  // int* n_features,
                                                 &n_classes,                   // int* n_classes,
                                                 &succeed,                     // int* succeed,  // -1 on error, 1 otherwise
                                                 model.NativePtr);             // IntPtr model);

                if (succeed != 1)
                    throw (new Exception("Unspecified error training SVM model"));

                model.NumClasses = n_classes;
                model.NumFeatures = n_features;
                model.Parameter = prms;

                return model;
            }
            catch
            {
                model.Dispose();
                throw;
            }
            finally
            {
                foreach (var hdl in gcHandles)
                {
                    if (hdl.IsAllocated) hdl.Free();
                }
                gcHandles.Clear();
            }
        }

        /// <summary>
        /// Train an SVM model on a dense dataset.
        /// </summary>
        /// <param name="prms">Calibration parameters</param>
        /// <param name="data">Feature vectors (stack as rows)</param>
        /// <param name="label"></param>
        /// <returns></returns>
        public static unsafe Model CreateSparse(Parameter prms, SparseMatrix data, float[] label)
        {
            if (data == null)
                throw (new ArgumentException("Data matrix is null"));
            data.Validate();

            if (label != null && data.RowCount != label.Length)
                throw (new ArgumentException("Number of rows in data matrix must match number of labels"));

            if (prms.LengthOfWeight > 0)
            {
                if (prms.Weight == null || prms.Weight.Length != prms.LengthOfWeight)
                    throw (new ArgumentException("LengthOfWeight does not match length of weight array"));

                if (prms.WeightLabel == null || prms.WeightLabel.Length != prms.LengthOfWeight)
                    throw (new ArgumentException("LengthOfWeight does not match length of weight label array"));
            }

            var gcHandles = new List<GCHandle>(6);

            var model = new Model(prms.SvmType);
            try
            {
                var dataHdl = GCHandle.Alloc(data.Data, GCHandleType.Pinned);
                gcHandles.Add(dataHdl);
                var dataPtr = (float*)dataHdl.AddrOfPinnedObject().ToPointer();

                var rowHdl = GCHandle.Alloc(data.RowExtents, GCHandleType.Pinned);
                gcHandles.Add(rowHdl);
                var rowPtr = (int*)rowHdl.AddrOfPinnedObject().ToPointer();

                var colHdl = GCHandle.Alloc(data.ColumnIndices, GCHandleType.Pinned);
                gcHandles.Add(colHdl);
                var colPtr = (int*)colHdl.AddrOfPinnedObject().ToPointer();

                float* labelPtr = null;
                if (label != null)
                {
                    var labelHdl = GCHandle.Alloc(label, GCHandleType.Pinned);
                    gcHandles.Add(labelHdl);
                    labelPtr = (float*)labelHdl.AddrOfPinnedObject().ToPointer();
                }

                int* weightLabelPtr = null;
                float* weightPtr = null;
                if (prms.LengthOfWeight > 0)
                {
                    var weightLabelHdl = GCHandle.Alloc(prms.WeightLabel, GCHandleType.Pinned);
                    gcHandles.Add(weightLabelHdl);
                    weightLabelPtr = (int*)weightLabelHdl.AddrOfPinnedObject().ToPointer();

                    var weightHdl = GCHandle.Alloc(prms.Weight, GCHandleType.Pinned);
                    gcHandles.Add(weightHdl);
                    weightPtr = (float*)weightHdl.AddrOfPinnedObject().ToPointer();
                }

                int n_features = 0;
                int n_classes = 0;
                int succeed = 0;
                NativeMethods.sparse_model_scikit(data.RowCount,                // int row_size, 
                                                  dataPtr,                      // float* val,
                                                  rowPtr,                       // int * row_ptr
                                                  colPtr,                       // int * col_ptr
                                                  labelPtr,                     // float* label,  // length: row_size (can also be null)
                                                  (int)prms.SvmType,            // int svm_type,
                                                  (int)prms.KernelType,         // int kernel_type,
                                                  prms.Degree,                  // int degree,
                                                  (float)prms.Gamma,            // float gamma,
                                                  (float)prms.Coef0,            // float coef0,
                                                  (float)prms.C,                // float cost,       // param_cmd.C = (float_type)cost;
                                                  (float)prms.Nu,               // float nu,
                                                  (float)prms.P,                // float epsilon,    // param_cmd.p = (float_type)epsilon;
                                                  (float)prms.Epsilon,          // float tol,        // param_cmd.epsilon = (float_type)tol;
                                                  (prms.Probability ? 1 : 0),   // int probability,
                                                  prms.LengthOfWeight,          // int weight_size,
                                                  weightLabelPtr,               // int* weight_label, // length: weight_size
                                                  weightPtr,                    // float* weight,     // length: weight_size
                                                  (prms.Verbose ? 1 : 0),       // int verbose,
                                                  prms.MaxIter,                 // int max_iter,
                                                  prms.NumCores,                // int n_cores,
                                                  prms.MaxMemSize,              // int max_mem_size,
                                                  &n_features,                  // int* n_features,
                                                  &n_classes,                   // int* n_classes,
                                                  &succeed,                     // int* succeed,  // -1 on error, 1 otherwise
                                                  model.NativePtr);             // IntPtr model);

                if (succeed != 1)
                    throw (new Exception("Unspecified error training SVM model"));

                model.NumClasses = n_classes;
                model.NumFeatures = n_features;
                model.Parameter = prms;

                return model;
            }
            catch
            {
                model.Dispose();
                throw;
            }
            finally
            {
                foreach (var hdl in gcHandles)
                {
                    if (hdl.IsAllocated) hdl.Free();
                }
                gcHandles.Clear();
            }
        }

        /// <summary>
        /// Calculate predicted labels for each of the given feature vectors.
        /// </summary>
        /// <param name="data">Input features (stacked as rows)</param>
        /// <param name="label">Output predicted labels</param>
        public unsafe void PredictDense(float[,] data, float[] label)
        {
            if (NativePtr == IntPtr.Zero)
                throw (new Exception("Model not initialised"));
            ThrowIfDisposed();

            if (data == null)
                throw (new ArgumentException("Data matrix is null"));

            if (label == null)
                throw (new ArgumentException("Label array is null"));

            if (data.GetLength(0) != label.Length)
                throw (new ArgumentException("Number of rows in data matrix must match number of labels"));

            if (data.GetLength(1) != NumFeatures)
                throw (new ArgumentException("Number of columns in data matrix must match number of features"));

            var gcHandles = new List<GCHandle>(2);

            try
            {
                var dataHdl = GCHandle.Alloc(data, GCHandleType.Pinned);
                gcHandles.Add(dataHdl);
                var dataPtr = (float*)dataHdl.AddrOfPinnedObject().ToPointer();

                var labelHdl = GCHandle.Alloc(label, GCHandleType.Pinned);
                gcHandles.Add(labelHdl);
                var labelPtr = (float*)labelHdl.AddrOfPinnedObject().ToPointer();

                NativeMethods.dense_predict(data.GetLength(0),            // int row_size, 
                                            data.GetLength(1),            // int features,
                                            dataPtr,                      // float* data,   // length: row_size * features (row-wise)
                                            NativePtr,                    // model
                                            labelPtr,                     // float* label,  // length: row_size (can also be null)
                                            (Parameter.Verbose ? 1 : 0)); // int verbose,
            }
            finally
            {
                foreach (var hdl in gcHandles)
                {
                    if (hdl.IsAllocated) hdl.Free();
                }
                gcHandles.Clear();
            }
        }

        /// <summary>
        /// Calculate predicted label for a given feature vector.
        /// </summary>
        /// <param name="data">Input features</param>
        public unsafe float PredictDense(float[] data)
        {
            if (NativePtr == IntPtr.Zero)
                throw (new Exception("Model not initialised"));
            ThrowIfDisposed();

            if (data == null)
                throw (new ArgumentException("Data matrix is null"));

            if (data.Length != NumFeatures)
                throw (new ArgumentException("Number of columns in data matrix must match number of features"));

            float label = 0;
            fixed (float *dataPtr = data)
            {
                NativeMethods.dense_predict(1,                            // int row_size, 
                                            data.Length,                  // int features,
                                            dataPtr,                      // float* data,   // length: row_size * features (row-wise)
                                            NativePtr,                    // model
                                            &label,                       // float* label,  // length: row_size (can also be null)
                                            (Parameter.Verbose ? 1 : 0)); // int verbose
            }
            return label;
        }

        /// <summary>
        /// Calculate predicted labels for each of the given feature vectors.
        /// </summary>
        /// <param name="data">Input features (stacked as rows)</param>
        /// <param name="label">Output predicted labels</param>
        public unsafe void PredictSparse(SparseMatrix data, float[] label)
        {
            if (NativePtr == IntPtr.Zero)
                throw (new Exception("Model not initialised"));
            ThrowIfDisposed();

            if (data == null)
                throw (new ArgumentException("Data matrix is null"));

            data.Validate();

            if (label == null)
                throw (new ArgumentException("Label array is null"));

            if (data.RowCount != label.Length)
                throw (new ArgumentException("Number of rows in data matrix must match number of labels"));

            if (data.ColumnCount != NumFeatures)
                throw (new ArgumentException("Number of columns in data matrix must match number of features"));

            fixed(float* dataPtr = data.Data, labelPtr = label)
            {
                fixed (int* rowPtr = data.RowExtents, colPtr = data.ColumnIndices)
                {
                    NativeMethods.sparse_predict(data.RowCount,                // int row_size, 
                                                 dataPtr,                      // float* val,
                                                 rowPtr,                       // int * row_ptr
                                                 colPtr,                       // int * col_ptr
                                                 NativePtr,                    // model
                                                 labelPtr,                     // float* label,  // length: row_size (can also be null)
                                                 (Parameter.Verbose ? 1 : 0)); // int verbose
                }
            }
        }

        /// <summary>
        /// Export model in format readable by LibSVM
        /// (Use WriteBinary/ReadBinary to persist Model)
        /// </summary>
        /// <param name="path"></param>
        public void Export(string path)
        {
            Save(path);
        }

        /// <summary>
        /// Saves this <see cref="Model"/> to the specified file.
        /// </summary>
        /// <param name="path">The file to write to.</param>
        private void Save(string path)
        {
            if (NativePtr == IntPtr.Zero)
                throw (new Exception("Model not initialised"));
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("The specified path is null or whitespace.");

            unsafe
            {
                NativeMethods.save_to_file_scikit(NativePtr, path);
            }
        }

        /// <summary>
        /// Loads an <see cref="Model"/> given the specified file.
        /// </summary>
        /// <param name="path">The LIBSVM format file name and path.</param>
        /// <returns>This method returns a new <see cref="Model"/> for the specified file.</returns>
        /// <exception cref="ArgumentException">The specified path is null or whitespace.</exception>
        /// <exception cref="FileNotFoundException">The specified file is not found.</exception>
        private static Model Load(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("The specified path is null or whitespace.");

            if (!File.Exists(path))
                throw new FileNotFoundException("The specified file is not found.");

            unsafe
            {
                var model = new Model(0);
                NativeMethods.load_from_file_scikit(model.NativePtr, path);
                return model;
            }
        }

        /// <summary>
        /// Persist model to binary stream
        /// </summary>
        /// <param name="writer"></param>
        public void WriteBinary(BinaryWriter writer)
        {
            if (NativePtr == IntPtr.Zero)
                throw (new Exception("Model not initialised"));
            ThrowIfDisposed();

            byte[] bytes = null;
            {
                var file = Path.GetTempFileName();
                try
                {
                    Save(file);
                    bytes = File.ReadAllBytes(file);
                }
                finally
                {
                    File.Delete(file);
                }
            }

            writer.Write(bytes.Length);
            writer.Write(bytes);
            writer.Write(NumClasses);
            writer.Write(NumFeatures);
            Parameter.WriteBinary(writer);
        }

        /// <summary>
        /// Load object from binary stream
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static Model ReadBinary(BinaryReader reader)
        {
            Model model = null;
            {
                var bytes = reader.ReadBytes(reader.ReadInt32());
                var file = Path.GetTempFileName();
                try
                {
                    File.WriteAllBytes(file, bytes);
                    model = Load(file);
                }
                finally
                {
                    File.Delete(file);
                }
            }
            try
            {
                model.NumClasses = reader.ReadInt32();
                model.NumFeatures = reader.ReadInt32();
                model.Parameter = Parameter.ReadBinary(reader);
                return model;
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Releases all unmanaged resources.
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            if (NativePtr != IntPtr.Zero)
                NativeMethods.model_free(NativePtr);
        }

        #endregion
    }
}
