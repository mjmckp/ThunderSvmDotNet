﻿using System;
using System.Runtime.InteropServices;
using System.Security;

// ReSharper disable InconsistentNaming
// ReSharper disable once CheckNamespace
namespace ThunderSvmDotNet.Interop
{

    internal static unsafe class NativeMethods
    {
        public const string NativeLibrary = @"x64\thundersvm.dll";

        public const CallingConvention CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl;

        #region Methods

        [SuppressUnmanagedCodeSecurity]
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern IntPtr model_new(int svm_type);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern void model_free(IntPtr model);


        [SuppressUnmanagedCodeSecurity]
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern void dense_model_scikit(int row_size,
                                                     int features,
                                                     float* data,   // length: row_size * features (row-wise)
                                                     float* label,  // length: row_size (can also be null)
                                                     int svm_type,
                                                     int kernel_type,
                                                     int degree,
                                                     float gamma,
                                                     float coef0,
                                                     float cost,
                                                     float nu,
                                                     float epsilon,
                                                     float tol,
                                                     int probability,
                                                     int weight_size,
                                                     int* weight_label, // length: weight_size
                                                     float* weight,     // length: weight_size
                                                     int verbose,
                                                     int max_iter,
                                                     int n_cores,
                                                     int max_mem_size,
                                                     int* n_features,
                                                     int* n_classes,
                                                     int* succeed,  // -1 on error, 1 otherwise
                                                     IntPtr model);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int dense_predict(int row_size, int features, float* data, IntPtr model, float* predict_label, int verbose);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern void save_to_file_scikit(IntPtr model, [MarshalAs(UnmanagedType.LPStr)]string path);

        [SuppressUnmanagedCodeSecurity]
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern void load_from_file_scikit(IntPtr model, [MarshalAs(UnmanagedType.LPStr)]string path);


#if false
        void dense_decision(int row_size, int features, float* data, SvmModel *model, int value_size, float* dec_value){

        void set_iter(SvmModel* model, int iter){

        void set_memory_size(SvmModel *model, int m_size){

        int n_sv(SvmModel* model){

        void get_sv(int* row, int* col, float* data, int* data_size, SvmModel* model){

        void get_support_classes(int* n_support, int n_class, SvmModel* model){

        void get_coef(float* dual_coef, int n_class, int n_sv, SvmModel* model){

        void get_rho(float* rho_, int rho_size, SvmModel* model){

        void get_pro(SvmModel *model, float* prob){

        void get_n_binary_models(SvmModel *model, int *n_model){

        void get_n_classes(SvmModel *model, int *n_classes){

        void sparse_model_scikit(int row_size, float* val, int* row_ptr, int* col_ptr, float* label,
                                      int svm_type, int kernel_type, int degree, float gamma, float coef0,
                                      float cost, float nu, float epsilon, float tol, int probability,
                                      int weight_size, int* weight_label, float* weight,
                                      int verbose, int max_iter, int n_cores, int max_mem_size,
                                      int* n_features, int* n_classes, int* succeed, SvmModel* model){

        int sparse_predict(int row_size, float* val, int* row_ptr, int* col_ptr, SvmModel *model, float* predict_label, int verbose){

        void sparse_decision(int row_size, float* val, int* row_ptr, int* col_ptr, SvmModel *model, int value_size, float* dec_value){
#endif

        #endregion
        }

    }
