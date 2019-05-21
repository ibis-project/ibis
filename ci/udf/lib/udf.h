// Copyright 2012 Cloudera Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef IMPALA_UDF_UDF_H
#define IMPALA_UDF_UDF_H

#include <assert.h>
#include <boost/cstdint.hpp>
#include <string.h>

/// This is the only Impala header required to develop UDFs and UDAs. This header
/// contains the types that need to be used and the FunctionContext object. The context
/// object serves as the interface object between the UDF/UDA and the impala process.
namespace impala {
  class FunctionContextImpl;
}

namespace impala_udf {

/// All input and output values will be one of the structs below. The struct is a simple
/// object containing a boolean to store if the value is NULL and the value itself. The
/// value is unspecified if the NULL boolean is set.
struct AnyVal;
struct BooleanVal;
struct TinyIntVal;
struct SmallIntVal;
struct IntVal;
struct BigIntVal;
struct StringVal;
struct TimestampVal;

/// A FunctionContext is passed to every UDF/UDA and is the interface for the UDF to the
/// rest of the system. It contains APIs to examine the system state, report errors and
/// manage memory.
class FunctionContext {
 public:
  enum ImpalaVersion {
    v1_2,
    v1_3,
  };

  enum Type {
    INVALID_TYPE = 0,
    TYPE_NULL,
    TYPE_BOOLEAN,
    TYPE_TINYINT,
    TYPE_SMALLINT,
    TYPE_INT,
    TYPE_BIGINT,
    TYPE_FLOAT,
    TYPE_DOUBLE,
    TYPE_TIMESTAMP,
    TYPE_STRING,
    TYPE_FIXED_BUFFER,
    TYPE_DECIMAL,
    TYPE_VARCHAR
  };

  struct TypeDesc {
    Type type;

    /// Only valid if type == TYPE_DECIMAL
    int precision;
    int scale;

    /// Only valid if type == TYPE_FIXED_BUFFER || type == TYPE_VARCHAR
    int len;
  };

  struct UniqueId {
    int64_t hi;
    int64_t lo;
  };

  enum FunctionStateScope {
    /// Indicates that the function state for this FunctionContext's UDF is shared across
    /// the plan fragment (a query is divided into multiple plan fragments, each of which
    /// is responsible for a part of the query execution). Within the plan fragment, there
    /// may be multiple instances of the UDF executing concurrently with multiple
    /// FunctionContexts sharing this state, meaning that the state must be
    /// thread-safe. The Prepare() function for the UDF may be called with this scope
    /// concurrently on a single host if the UDF will be evaluated in multiple plan
    /// fragments on that host. In general, read-only state that doesn't need to be
    /// recomputed for every UDF call should be fragment-local.
    /// TODO: not yet implemented
    FRAGMENT_LOCAL,

    /// Indicates that the function state is local to the execution thread. This state
    /// does not need to be thread-safe. However, this state will be initialized (via the
    /// Prepare() function) once for every execution thread, so fragment-local state
    /// should be used when possible for better performance. In general, inexpensive
    /// shared state that is written to by the UDF (e.g. scratch space) should be
    /// thread-local.
    THREAD_LOCAL,
  };

  /// Returns the version of Impala that's currently running.
  ImpalaVersion version() const;

  /// Returns the user that is running the query. Returns NULL if it is not
  /// available.
  const char* user() const;

  /// Returns the effective user for authorization purposes. If a delegated user is
  /// configured, returns that user, otherwise returns the same as user().
  const char* effective_user() const;

  /// Returns the query_id for the current query.
  UniqueId query_id() const;

  /// Sets an error for this UDF. If this is called, this will trigger the
  /// query to fail.
  void SetError(const char* error_msg);

  /// Adds a warning that is returned to the user. This can include things like
  /// overflow or other recoverable error conditions.
  /// Warnings are capped at a maximum number. Returns true if the warning was
  /// added and false if it was ignored due to the cap.
  bool AddWarning(const char* warning_msg);

  /// Returns true if there's been an error set.
  bool has_error() const;

  /// Returns the current error message. Returns NULL if there is no error.
  const char* error_msg() const;

  /// Allocates memory. All UDF/UDAs should use this if possible instead of malloc/new.
  /// The UDF/UDA is responsible for calling Free() on all buffers returned by Allocate().
  /// If Allocate() fails or causes the memory limit to be exceeded, the error will be
  /// set in this object causing the query to fail.
  uint8_t* Allocate(int byte_size);

  /// Wrapper around Allocate() to allocate a buffer of the given type "T".
  template<typename T>
  T* Allocate() {
    return reinterpret_cast<T*>(Allocate(sizeof(T)));
  }

  /// Reallocates 'ptr' to the new byte_size. If the currently underlying allocation
  /// is big enough, the original ptr will be returned. If the allocation needs to
  /// grow, a new allocation is made that is at least 'byte_size' and the contents
  /// of 'ptr' will be copied into it. If the new allocation fails or causes the
  /// memory limit to be exceeded, the error will be set in this object.
  ///
  /// This should be used for buffers that constantly get appended to.
  uint8_t* Reallocate(uint8_t* ptr, int byte_size);

  /// Frees a buffer returned from Allocate() or Reallocate()
  void Free(uint8_t* buffer);

  /// For allocations that cannot use the Allocate() API provided by this
  /// object, TrackAllocation()/Free() can be used to just keep count of the
  /// byte sizes. For each call to TrackAllocation(), the UDF/UDA must call
  /// the corresponding Free().
  void TrackAllocation(int64_t byte_size);
  void Free(int64_t byte_size);

  /// Methods for maintaining state across UDF/UDA function calls. SetFunctionState() can
  /// be used to store a pointer that can then be retreived via GetFunctionState(). If
  /// GetFunctionState() is called when no pointer is set, it will return
  /// NULL. SetFunctionState() does not take ownership of 'ptr'; it is up to the UDF/UDA
  /// to clean up any function state if necessary.
  void SetFunctionState(FunctionStateScope scope, void* ptr);
  void* GetFunctionState(FunctionStateScope scope) const;

  /// Returns the return type information of this function. For UDAs, this is the final
  /// return type of the UDA (e.g., the type returned by the finalize function).
  const TypeDesc& GetReturnType() const;

  /// Returns the intermediate type for UDAs, i.e., the one returned by
  /// update and merge functions. Returns INVALID_TYPE for UDFs.
  const TypeDesc& GetIntermediateType() const;

  /// Returns the number of arguments to this function (not including the FunctionContext*
  /// argument).
  int GetNumArgs() const;

  /// Returns the type information for the arg_idx-th argument (0-indexed, not including
  /// the FunctionContext* argument). Returns NULL if arg_idx is invalid.
  const TypeDesc* GetArgType(int arg_idx) const;

  /// Returns true if the arg_idx-th input argument (0 indexed, not including the
  /// FunctionContext* argument) is a constant (e.g. 5, "string", 1 + 1).
  bool IsArgConstant(int arg_idx) const;

  /// Returns a pointer to the value of the arg_idx-th input argument (0 indexed, not
  /// including the FunctionContext* argument). Returns NULL if the argument is not
  /// constant. This function can be used to obtain user-specified constants in a UDF's
  /// Init() or Close() functions.
  AnyVal* GetConstantArg(int arg_idx) const;

  /// TODO: Do we need to add arbitrary key/value metadata. This would be plumbed
  /// through the query. E.g. "select UDA(col, 'sample=true') from tbl".
  /// const char* GetMetadata(const char*) const;

  /// TODO: Add mechanism for UDAs to update stats similar to runtime profile counters

  /// TODO: Add mechanism to query for table/column stats

  /// Returns the underlying opaque implementation object. The UDF/UDA should not
  /// use this. This is used internally.
  impala::FunctionContextImpl* impl() { return impl_; }

  ~FunctionContext();

 private:
  friend class impala::FunctionContextImpl;
  FunctionContext();

  /// Disable copy ctor and assignment operator
  FunctionContext(const FunctionContext& other);
  FunctionContext& operator=(const FunctionContext& other);

  impala::FunctionContextImpl* impl_; // Owned by this object.
};

//----------------------------------------------------------------------------
//------------------------------- UDFs ---------------------------------------
//----------------------------------------------------------------------------
/// The UDF must implement this function prototype. This is not a typedef as the actual
/// UDF's signature varies from UDF to UDF.
///    typedef <*Val> Evaluate(FunctionContext* context, <const Val& arg>);
///
/// The UDF must return one of the *Val structs. The UDF must accept a pointer to a
/// FunctionContext object and then a const reference for each of the input arguments.
/// Examples of valid Udf signatures are:
///  1) DoubleVal Example1(FunctionContext* context);
///  2) IntVal Example2(FunctionContext* context, const IntVal& a1, const DoubleVal& a2);
///
/// UDFs can be variadic. The variable arguments must all come at the end and must be
/// the same type. A example signature is:
///  StringVal Concat(FunctionContext* context, const StringVal& separator,
///    int num_var_args, const StringVal* args);
/// In this case args[0] is the first variable argument and args[num_var_args - 1] is
/// the last.
///
/// ------- Memory Management -------
/// ---------------------------------
/// The UDF can assume that memory from input arguments will have the same lifetime as
/// results for the UDF. In other words, the UDF can return memory from input arguments
/// without making copies. For example, a function like substring will not need to
/// allocate and copy the smaller string.
///
/// Any state needed across calls must be stored and accessed via
/// FunctionContext::SetFunctionState() and FunctionContext::GetFunctionState(). The UDF
/// should not maintain any other state across calls since there is no guarantee on how
/// the execution is multithreaded or distributed.
///
/// For StringVal return values, the UDF can use StringVal(FunctionContext*, int)
/// ctor or the function StringVal::CopyFrom(FunctionContext*, const uint8_t*, size_t).
/// The memory consumed by the StringVal will be managed by Impala. Please see the UDA
/// section below for details.
///
/// -------- Execution Model --------
/// ---------------------------------
/// Execution model: For each UDF use occurring in a given query, at least one
/// FunctionContext will be created. For a given FunctionContext, the UDF's functions are
/// never called concurrently and therefore do not need to be thread-safe. State shared
/// across UDF invocations should be initialized and cleaned up using prepare and close
/// functions (described below).
///
/// Note that a single UDF use may produce multiple FunctionContexts for that UDF (this is
/// so the UDF can be executed concurrently in different threads). For example, the query
/// "select * from tbl where my_udf(x) > 0" may produce multiple FunctionContexts for
/// 'my_udf', each of which may concurrently be passed to 'my_udf's prepare, close, and
/// UDF functions.
///
/// --- Prepare / Close Functions ---
/// ---------------------------------
/// The UDF can optionally include a prepare function, specified in the "CREATE FUNCTION"
/// statement using "prepare_fn=<prepare function symbol>". The prepare function is called
/// before any calls to the UDF to evaluate values. This is the appropriate time for the
/// UDF to initialize any shared data structures, validate versions, etc. If there is an
/// error, this function should call FunctionContext::SetError()/
/// FunctionContext::AddWarning().
///
/// The prepare function is called multiple times with different FunctionStateScopes. It
/// will be called once per fragment with 'scope' set to FRAGMENT_LOCAL, and once per
/// execution thread with 'scope' set to THREAD_LOCAL.
typedef void (*UdfPrepare)(FunctionContext* context,
                           FunctionContext::FunctionStateScope scope);

/// The UDF can also optionally include a close function, specified in the "CREATE
/// FUNCTION" statement using "close_fn=<close function symbol>". The close function is
/// called after all calls to the UDF have completed. This is the appropriate time for the
/// UDF to deallocate any shared data structures that are not needed to maintain the
/// results. If there is an error, this function should call FunctionContext::SetError()/
/// FunctionContext::AddWarning().
//
/// The close function is called multiple times with different FunctionStateScopes. It
/// will be called once per fragment with 'scope' set to FRAGMENT_LOCAL, and once per
/// execution thread with 'scope' set to THREAD_LOCAL.
typedef void (*UdfClose)(FunctionContext* context,
                         FunctionContext::FunctionStateScope scope);

//----------------------------------------------------------------------------
//------------------------------- UDAs ---------------------------------------
//----------------------------------------------------------------------------
/// The UDA execution is broken up into a few steps. The general calling pattern
/// is one of these:
///  1) Init(), Update() (repeatedly), Serialize()
///  2) Init(), Update() (repeatedly), Finalize()
///  3) Init(), Merge() (repeatedly), Serialize()
///  4) Init(), Merge() (repeatedly), Finalize()
/// The UDA is registered with three types: the result type, the input type and
/// the intermediate type.
///
/// If the UDA needs a fixed byte width intermediate buffer, the type should be
/// TYPE_FIXED_BUFFER and Impala will allocate the buffer. If the UDA needs an unknown
/// sized buffer, it should use TYPE_STRING and allocate it from the FunctionContext
/// manually.
/// For UDAs that need a complex data structure as the intermediate state, the
/// intermediate type should be string and the UDA can cast the ptr to the structure
/// it is using.
///
/// Memory Management: For allocations that are not returned to Impala, the UDA should use
/// the FunctionContext::Allocate()/Free() methods. In general, Allocate() is called in
/// Init(), and then Free() must be called in both Serialize() and Finalize(), since
/// either of these functions may be called to clean up the state. For StringVal
/// allocations returned to Impala (e.g. returned by UdaSerialize()), the UDA should
/// allocate the result via StringVal(FunctionContext*, int) ctor or the function
/// StringVal::CopyFrom(FunctionContext*, const uint8_t*, size_t) and Impala will
/// automatically handle freeing it.
//
/// Note that in the rare case the StringVal ctor or StringVal::CopyFrom() fail to
/// allocate memory, the StringVal object will be marked as a null string.
/// Serialize()/Finalize() should handle allocation failures by checking the is_null
/// field of the StringVal object and carry out appropriate error handling action.
/// Similarly, FunctionContext::Allocate()/Reallocate() may also fail to allocate
/// memory so callers should check the returned values before using them.
///
/// For clarity in documenting the UDA interface, the various types will be typedefed
/// here. The actual execution resolves all the types at runtime and none of these types
/// should actually be used.
///
/// TODO: add an Init() variant that takes the initial input value to avoid initializing
/// then immediately overwriting the value.
typedef AnyVal InputType;
typedef AnyVal InputType2;
typedef AnyVal ResultType;
typedef AnyVal IntermediateType;

/// UdaInit is called once for each aggregate group before calls to any of the
/// other functions below.
typedef void (*UdaInit)(FunctionContext* context, IntermediateType* result);

/// This is called for each input value. The UDA should update result based on the
/// input value. The update function can take any number of input arguments. Here
/// are some examples:
typedef void (*UdaUpdate)(FunctionContext* context, const InputType& input,
    IntermediateType* result);
typedef void (*UdaUpdate2)(FunctionContext* context, const InputType& input,
    const InputType2& input2, IntermediateType* result);

/// Merge an intermediate result 'src' into 'dst'.
typedef void (*UdaMerge)(FunctionContext* context, const IntermediateType& src,
    IntermediateType* dst);

/// Serialize the intermediate type. The serialized data is then sent across the
/// wire.
/// No additional functions will be called with this FunctionContext object and the
/// UDA should do final clean (e.g. Free()) here.
typedef const IntermediateType (*UdaSerialize)(FunctionContext* context,
    const IntermediateType& type);

/// Called once at the end to return the final value for this UDA.
/// No additional functions will be called with this FunctionContext object and the
/// UDA should do final clean (e.g. Free()) here.
typedef ResultType (*UdaFinalize)(FunctionContext* context, const IntermediateType& v);

//----------------------------------------------------------------------------
//-------------Implementation of the *Val structs ----------------------------
//----------------------------------------------------------------------------
struct AnyVal {
  bool is_null;
  AnyVal(bool is_null = false) : is_null(is_null) {}
};

struct BooleanVal : public AnyVal {
  bool val;

  BooleanVal(bool val = false) : val(val) {}

  static BooleanVal null() {
    BooleanVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const BooleanVal& other) const {
    if (is_null && other.is_null) return true;
    if (is_null || other.is_null) return false;
    return val == other.val;
  }
  bool operator!=(const BooleanVal& other) const { return !(*this == other); }
};

struct TinyIntVal : public AnyVal {
  int8_t val;

  TinyIntVal(int8_t val = 0) : val(val) { }

  static TinyIntVal null() {
    TinyIntVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const TinyIntVal& other) const {
    if (is_null && other.is_null) return true;
    if (is_null || other.is_null) return false;
    return val == other.val;
  }
  bool operator!=(const TinyIntVal& other) const { return !(*this == other); }
};

struct SmallIntVal : public AnyVal {
  int16_t val;

  SmallIntVal(int16_t val = 0) : val(val) { }

  static SmallIntVal null() {
    SmallIntVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const SmallIntVal& other) const {
    if (is_null && other.is_null) return true;
    if (is_null || other.is_null) return false;
    return val == other.val;
  }
  bool operator!=(const SmallIntVal& other) const { return !(*this == other); }
};

struct IntVal : public AnyVal {
  int32_t val;

  IntVal(int32_t val = 0) : val(val) { }

  static IntVal null() {
    IntVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const IntVal& other) const {
    if (is_null && other.is_null) return true;
    if (is_null || other.is_null) return false;
    return val == other.val;
  }
  bool operator!=(const IntVal& other) const { return !(*this == other); }
};

struct BigIntVal : public AnyVal {
  int64_t val;

  BigIntVal(int64_t val = 0) : val(val) { }

  static BigIntVal null() {
    BigIntVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const BigIntVal& other) const {
    if (is_null && other.is_null) return true;
    if (is_null || other.is_null) return false;
    return val == other.val;
  }
  bool operator!=(const BigIntVal& other) const { return !(*this == other); }
};

struct FloatVal : public AnyVal {
  float val;

  FloatVal(float val = 0) : val(val) { }

  static FloatVal null() {
    FloatVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const FloatVal& other) const {
    return is_null == other.is_null && val == other.val;
  }
  bool operator!=(const FloatVal& other) const { return !(*this == other); }
};

struct DoubleVal : public AnyVal {
  double val;

  DoubleVal(double val = 0) : val(val) { }

  static DoubleVal null() {
    DoubleVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const DoubleVal& other) const {
    if (is_null && other.is_null) return true;
    if (is_null || other.is_null) return false;
    return val == other.val;
  }
  bool operator!=(const DoubleVal& other) const { return !(*this == other); }
};

/// This object has a compatible storage format with boost::ptime.
struct TimestampVal : public AnyVal {
  /// Gregorian date. This has the same binary format as boost::gregorian::date.
  int32_t date;
  /// Nanoseconds in current day.
  int64_t time_of_day;

  TimestampVal(int32_t date = 0, int64_t time_of_day = 0) :
    date(date), time_of_day(time_of_day) {
  }

  static TimestampVal null() {
    TimestampVal result;
    result.is_null = true;
    return result;
  }

  bool operator==(const TimestampVal& other) const {
    if (is_null && other.is_null) return true;
    if (is_null || other.is_null) return false;
    return date == other.date && time_of_day == other.time_of_day;
  }
  bool operator!=(const TimestampVal& other) const { return !(*this == other); }
};

/// Note: there is a difference between a NULL string (is_null == true) and an
/// empty string (len == 0).
struct StringVal : public AnyVal {

  static const int MAX_LENGTH = (1 << 30);

  int len;
  uint8_t* ptr;

  /// Construct a StringVal from ptr/len. Note: this does not make a copy of ptr
  /// so the buffer must exist as long as this StringVal does.
  StringVal(uint8_t* ptr = NULL, int len = 0) : len(len), ptr(ptr) {
    assert(len >= 0);
    if (ptr == NULL) assert(len == 0);
  };

  /// Construct a StringVal from NULL-terminated c-string. Note: this does not make a
  /// copy of ptr so the underlying string must exist as long as this StringVal does.
  StringVal(const char* ptr) : len(strlen(ptr)), ptr((uint8_t*)ptr) {}

  /// Creates a StringVal, allocating a new buffer with 'len'. This should
  /// be used to return StringVal objects in UDF/UDAs that need to allocate new
  /// string memory.
  ///
  /// If the memory allocation fails, e.g. because the intermediate value would be too
  /// large, the constructor will construct a NULL string and set an error on the function
  /// context.
  StringVal(FunctionContext* context, int len);

  /// Will create a new StringVal with the given dimension and copy the data from the
  /// parameters. In case of an error will return a NULL string and set an error on the
  /// function context.
  ///
  /// Note that the memory for the buffer of the new StringVal is managed by Impala.
  /// Impala will handle freeing it. Callers should not call Free() on the 'ptr' of
  /// the StringVal returned.
  static StringVal CopyFrom(FunctionContext* ctx, const uint8_t* buf, size_t len);

  static StringVal null() {
    StringVal sv;
    sv.is_null = true;
    return sv;
  }

  bool operator==(const StringVal& other) const {
    if (is_null != other.is_null) return false;
    if (is_null) return true;
    if (len != other.len) return false;
    return ptr == other.ptr || memcmp(ptr, other.ptr, len) == 0;
  }
  bool operator!=(const StringVal& other) const { return !(*this == other); }
};

struct DecimalVal : public impala_udf::AnyVal {
  /// Decimal data is stored as an unscaled integer value. For example, the decimal 1.00
  /// (precison 3, scale 2) is stored as 100. The byte size necessary to store the decimal
  /// depends on the precision, which determines which field of the union should be used to
  /// store and manipulate the unscaled value.
  //
  ///   precision between 0-9:   val4  (4 bytes)
  ///   precision between 10-18: val8  (8 bytes)
  ///   precision between 19-38: val16 (16 bytes)
  //
  /// While it is always safe to use a larger field than necessary, it may result in worse
  /// performance. For example, a UDF that only uses val16 can handle any precision but may
  /// be slower than one that uses val4 or val8.
  union {
    int32_t val4;
    int64_t val8;
    __int128_t val16;
  };

  DecimalVal() : val16(0) {}
  DecimalVal(int32_t v) : val16(v) {}
  DecimalVal(int64_t v) : val16(v) {}
  DecimalVal(__int128_t v) : val16(v) {}

  static DecimalVal null() {
    DecimalVal result;
    result.is_null = true;
    return result;
  }

  DecimalVal& operator=(const DecimalVal& other) {
    // Depending on the compiler, the default assignment operator may require 16-byte
    // alignment of 'this' and 'other'. Cast to void* so the compiler doesn't change back
    // to an assignment.
    memcpy(reinterpret_cast<void*>(this), reinterpret_cast<const void*>(&other),
           sizeof(DecimalVal));
    return *this;
  }

  DecimalVal(const DecimalVal& other) {
    *this = other;
  }
};

typedef uint8_t* BufferVal;

}

#endif
