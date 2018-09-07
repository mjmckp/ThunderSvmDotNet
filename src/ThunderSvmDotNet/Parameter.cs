using System;
using System.Runtime.InteropServices;
using ThunderSvmDotNet.Interop;

namespace ThunderSvmDotNet
{

    /// <summary>
    /// Represents an parameter for Support Vector Machine.
    /// </summary>
    public sealed class Parameter
    {

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="Parameter"/> class.
        /// </summary>
        public Parameter(int numFeatures)
        {
            // defaults copied from svmparam.h
            SvmType = SvmType.CSVC;
            KernelType = KernelType.RBF;
            C = 1;
            Gamma = (numFeatures > 0) ? (1.0 /(double)numFeatures) : 0.0;
            P = 0.1;
            Epsilon = 0.001;
            Nu = 0.5;
            Probability = false;
            LengthOfWeight = 0;
            Degree = 3;
            Coef0 = 0;
            Verbose = false;
            MaxMemSize = -1;
            MaxIter = -1;
            NumCores = -1;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets the cost parameter.
        /// </summary>
        public double C
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the maximum memory size in bytes (set to -1 for default value of 8192 &lt;&lt; 20)
        /// </summary>
        public int MaxMemSize
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the coef0 parameter in kernel function.
        /// </summary>
        public double Coef0
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the degree parameter in kernel function.
        /// </summary>
        public int Degree
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the tolerance of termination criterion.
        /// </summary>
        public double Epsilon
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the gamma parameter in kernel function.
        /// </summary>
        public double Gamma
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the type of kernel function.
        /// </summary>
        public KernelType KernelType
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the number of elements in the array <see cref="Weight"/> and <see cref="WeightLabel"/>.
        /// </summary>
        public int LengthOfWeight
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the nu parameter of SvmType.NuSVC, SvmType.OneClass, and SvmType.NuSVR.
        /// </summary>
        public double Nu
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the epsilon in loss function of SvmType.EpsilonSVR.
        /// </summary>
        public double P
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets indicating whether to train a SVC or SVR model for probability estimates.
        /// </summary>
        public bool Probability
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets type of Support Vector Machine.
        /// </summary>
        public SvmType SvmType
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the array of factors to change penalty for some class.
        /// </summary>
        /// <remarks>Each <code>Weight[i]</code> corresponds to <code>WeightLabel[i]</code>.</remarks>
        public float[] Weight
        {
            get;
            set;
        }

        /// <summary>
        /// Gets or sets the array of labels to change penalty for some class.
        /// </summary>
        /// <remarks>Each <code>Weight[i]</code> corresponds to <code>WeightLabel[i]</code>.</remarks>
        public int[] WeightLabel
        {
            get;
            set;
        }

        /// <summary>
        /// Logging verbosity.
        /// </summary>
        public bool Verbose
        {
            get;
            set;
        }

        /// <summary>
        /// Hard limit on iterations within solver, or -1 for no limit.
        /// </summary>
        public int MaxIter
        {
            get;
            set;
        }

        /// <summary>
        /// Limit of number of OMP threads, or -1 to set to maximum.
        /// </summary>
        public int NumCores
        {
            get;
            set;
        }

        #endregion

        /// <summary>
        /// Persist object to binary stream
        /// </summary>
        /// <param name="writer"></param>
        public void WriteBinary(System.IO.BinaryWriter writer)
        {
            writer.Write(C);
            writer.Write(MaxMemSize);
            writer.Write(Coef0);
            writer.Write(Degree);
            writer.Write(Epsilon);
            writer.Write(Gamma);
            writer.Write((int)KernelType);
            writer.Write(Nu);
            writer.Write(P);
            writer.Write(Probability);
            writer.Write((int)SvmType);
            writer.Write(Verbose);
            writer.Write(MaxIter);
            writer.Write(NumCores);
            writer.Write(LengthOfWeight);
            if (LengthOfWeight > 0)
            {
                foreach(var w in Weight)
                    writer.Write(w);
                foreach (var wl in WeightLabel)
                    writer.Write(wl);
            }
        }

        /// <summary>
        /// Load object from binary stream
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static Parameter ReadBinary(System.IO.BinaryReader reader)
        {
            var rslt = new Parameter(0)
            {
                C = reader.ReadDouble(),
                MaxMemSize = reader.ReadInt32(),
                Coef0 = reader.ReadDouble(),
                Degree = reader.ReadInt32(),
                Epsilon = reader.ReadDouble(),
                Gamma = reader.ReadDouble(),
                KernelType = (KernelType)reader.ReadInt32(),
                Nu = reader.ReadDouble(),
                P = reader.ReadDouble(),
                Probability = reader.ReadBoolean(),
                SvmType = (SvmType)reader.ReadInt32(),
                Verbose = reader.ReadBoolean(),
                MaxIter = reader.ReadInt32(),
                NumCores = reader.ReadInt32(),
                LengthOfWeight = reader.ReadInt32()
            };
            if (rslt.LengthOfWeight > 0)
            {
                rslt.Weight = new float[rslt.LengthOfWeight];
                rslt.WeightLabel = new int[rslt.LengthOfWeight];
                for (var i = 0; i < rslt.Weight.Length; i++)
                    rslt.Weight[i] = reader.ReadSingle();
                for (var i = 0; i < rslt.WeightLabel.Length; i++)
                    rslt.WeightLabel[i] = reader.ReadInt32();
            }
            return rslt;
        }

    }

}