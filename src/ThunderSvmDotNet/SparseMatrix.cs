using System;

namespace ThunderSvmDotNet
{
    /// <summary>
    /// Sparse matrix in compressed sparse row (CSR) format
    /// </summary>
    public class SparseMatrix
    {
        /// <summary>
        /// Empty constructor
        /// </summary>
        public SparseMatrix()
        {
        }

        /// <summary>
        /// Constructor with known dimensions
        /// </summary>
        public SparseMatrix(int rowCount, int columnCount, int numNonZeroValues)
        {
            ColumnCount = columnCount;
            Data = new float[numNonZeroValues];
            ColumnIndices = new int[numNonZeroValues];
            RowExtents = new int[rowCount + 1];
        }

        /// <summary>
        /// Array of all nonzero entries in matrix, in left-to-right top-to-bottom ("row-major") order.
        /// </summary>
        public float [] Data
        {
            get; set;
        }

        /// <summary>
        /// Array of length M+1 (where M is number of rows in the matrix), such that
        /// the data in the i-th row of the matrix is in Data[RowExtents[i]] to Data[RowExtents[i+1]].
        /// </summary>
        public int[] RowExtents
        {
            get; set;
        }

        /// <summary>
        /// Array containing column index corresponding to each entry the in the Data array.
        /// </summary>
        public int[] ColumnIndices
        {
            get; set;
        }

        /// <summary>
        /// Number of matrix columns
        /// </summary>
        public int ColumnCount
        {
            get; set;
        }

        /// <summary>
        /// Number of matrix rows
        /// </summary>
        public int RowCount => RowExtents.Length - 1;

        /// <summary>
        /// Check validity of the sparse matrix representation
        /// </summary>
        public void Validate()
        {
            if (Data == null) throw (new ArgumentNullException("Data"));
            if (RowExtents == null) throw (new ArgumentNullException("RowExtents"));
            if (ColumnIndices == null) throw (new ArgumentNullException("ColumnIndices"));

            if (ColumnCount < 0) throw (new Exception("ColumnCount must be non-negative"));
            if (Data.Length != ColumnIndices.Length) throw (new Exception("ColumnIndices.Length must match Data.Length"));
            foreach(var index in ColumnIndices)
            {
                if (!(0 <= index && index < ColumnCount))
                    throw (new Exception($"Invalid Column Index: {index}"));
            }

            if (RowExtents.Length == 0) throw (new Exception("RowExtents.Length must be > 0"));
            if (RowExtents[0] != 0) throw (new Exception("RowExtents[0] must be 0"));
            if (RowExtents[RowExtents.Length-1] != Data.Length) throw (new Exception("RowExtents[RowExtents.Length-1] must be equal to Data.Length"));
            for (var i = 1; i < RowExtents.Length - 1; i++)
            {
                if (!(RowExtents[i - 1] <= RowExtents[i]))
                    throw (new Exception("RowExtents must be non-decreasing."));
            }
        }
    }
}
