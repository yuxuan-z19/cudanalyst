#pragma once

#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define PROFILING_TOTAL_TIME (0)
#define PROFILING_CREATE (1)
#define PROFILING_RANK (2)
#define PROFILING_VERIFY (3)

/*****************************************************************/
/* for serial IS, buckets are not really req'd to solve NPB1 IS  */
/* spec, but their use on some machines improves performance, on */
/* other machines the use of buckets compromises performance,    */
/* probably because it is extra computation which is not req'd.  */
/* (note: mechanism not understood, probably cache related)      */
/* example: SP2-66MhzWN: 50% speedup with buckets                */
/* example: SGI Indy5000: 50% slowdown with buckets              */
/* example: SGI O2000: 400% slowdown with buckets (Wow!)         */
/*****************************************************************/
/* to disable the use of buckets, comment out the following line */
#define USE_BUCKETS

/******************/
/* default values */
/******************/
#ifndef CLASS
#define CLASS 'S'
#endif

/*************/
/*  CLASS S  */
/*************/
#if CLASS == 'S'
#define TOTAL_KEYS_LOG_2 (16)
#define MAX_KEY_LOG_2 (11)
#define NUM_BUCKETS_LOG_2 (9)
#endif

/*************/
/*  CLASS W  */
/*************/
#if CLASS == 'W'
#define TOTAL_KEYS_LOG_2 (20)
#define MAX_KEY_LOG_2 (16)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS A  */
/*************/
#if CLASS == 'A'
#define TOTAL_KEYS_LOG_2 (23)
#define MAX_KEY_LOG_2 (19)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS B  */
/*************/
#if CLASS == 'B'
#define TOTAL_KEYS_LOG_2 (25)
#define MAX_KEY_LOG_2 (21)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS C  */
/*************/
#if CLASS == 'C'
#define TOTAL_KEYS_LOG_2 (27)
#define MAX_KEY_LOG_2 (23)
#define NUM_BUCKETS_LOG_2 (10)
#endif

/*************/
/*  CLASS D  */
/*************/
#if CLASS == 'D'
#define TOTAL_KEYS_LOG_2 (31)
#define MAX_KEY_LOG_2 (27)
#define NUM_BUCKETS_LOG_2 (10)
#endif

#if CLASS == 'D'
#define TOTAL_KEYS (1L << TOTAL_KEYS_LOG_2)
#else
#define TOTAL_KEYS (1 << TOTAL_KEYS_LOG_2)
#endif
#define MAX_KEY (1 << MAX_KEY_LOG_2)
#define NUM_BUCKETS (1 << NUM_BUCKETS_LOG_2)
#define NUM_KEYS (TOTAL_KEYS)
#define SIZE_OF_BUFFERS (NUM_KEYS)

#define MAX_ITERATIONS (10)
#define TEST_ARRAY_SIZE (5)

/*************************************/
/* typedef: if necessary, change the */
/* size of INT_TYPE here by changing the  */
/* INT_TYPE type to, say, long            */
/*************************************/
#if CLASS == 'D'
typedef long INT_TYPE;
#else
typedef int INT_TYPE;
#endif

extern __shared__ INT_TYPE extern_share_data[];
