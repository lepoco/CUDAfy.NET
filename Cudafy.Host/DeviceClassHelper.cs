/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2011 Hybrid DSP Systems
http://www.hybriddsp.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Reflection.Emit;
using Cudafy.Host;
using Cudafy;

namespace Cudafy.Host
{
    /// <summary>
    /// For a class, and instance of which we want to move onto the Device, we create a new representation of the instance on the device.
    /// This is the DeviceType, which basically has reference type fields replaced by IntPtrs to device memory.
    /// </summary>
    public class DeviceTypeInfo
    {
        public Type DeviceType;
        public List<FieldMapping> PointerFields;
        public List<FieldMapping> NonPointerFields;
    }

    /// <summary>
    /// Helper that can create an instance of a class or struct on the device, given an instance of
    /// a Cudafyable class on the CPU.
    /// Any reference type members of the class become pointers to the member (in device memory). 
    /// Any value type members must be blittable.
    /// A one-to-one map of objects on the host and objects on the device is maintained so that instances that share an array on
    /// the host share the same array on the device.
    /// </summary>
    public class DeviceClassHelper
    {
        static ModuleBuilder moduleBuilder;

        static Dictionary<Type, DeviceTypeInfo> deviceTypeLookup = new Dictionary<Type, DeviceTypeInfo>();

        /// <summary>
        /// This provides a mapping of the host object to the object that encapsulates the device pointer (via _deviceMemory).
        /// </summary>
        static Dictionary<GPGPU, Dictionary<object, object>> deviceObjectFromHostObject = new Dictionary<GPGPU, Dictionary<object, object>>();

        /// <summary>
        /// Returns the device object that is mapped to the host object.
        /// </summary>
        /// <param name="gpu"></param>
        /// <param name="hostObject"></param>
        /// <returns></returns>
        public static object TryGetDeviceObjectFromHostObject(GPGPU gpu, object hostObject)
        {
            object deviceObject;
            if (!deviceObjectFromHostObject[gpu].TryGetValue(hostObject, out deviceObject)) return null;
            else return deviceObject;
        }

        /// <summary>
        /// Returns the device pointer for the device object that is mapped to the host object.
        /// </summary>
        /// <param name="gpu"></param>
        /// <param name="hostObject"></param>
        /// <returns></returns>
        public static DevicePtrEx TryGetDeviceMemoryFromHostObject(GPGPU gpu, object hostObject)
        {
            object deviceObject;
            if (!deviceObjectFromHostObject[gpu].TryGetValue(hostObject, out deviceObject)) return null;
            else return gpu.TryGetDeviceMemory(deviceObject);
        }

        static DeviceClassHelper()
        {
            AssemblyName assemblyName = new AssemblyName("DynamicTypesAssembly");
            AssemblyBuilder ab =
                AppDomain.CurrentDomain.DefineDynamicAssembly(
                    assemblyName,
                    AssemblyBuilderAccess.RunAndSave);

            moduleBuilder =
                ab.DefineDynamicModule(assemblyName.Name, assemblyName.Name + ".dll");
        }

        /// <summary>
        /// Creates a version of hostObject on the device with any fields which are reference types converted
        /// to pointers to device memory (IntPtrs) and any arrays copied to the device.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="gpu"></param>
        /// <param name="hostObject"></param>
        /// <returns>THe device object (that can then be used in kernal calls).</returns>
        public static object CreateDeviceObject<T>(GPGPU gpu, T hostObject)
        {
            if (!deviceObjectFromHostObject.ContainsKey(gpu))
                deviceObjectFromHostObject.Add(gpu, new Dictionary<object, object>());

            Type hostObjectType = hostObject.GetType();
            Type newTypeToCreate;
            if (hostObjectType.IsArray)
            {
                newTypeToCreate = hostObjectType.GetElementType();
                // if this is a value type array
                if (newTypeToCreate.IsValueType)
                    return CopyArrayToDevice(gpu, hostObject as Array);
                else
                    return CopyArrayOfReferenceTypeToDevice(gpu, hostObject as Array);
            }
            else
                newTypeToCreate = hostObjectType;

            // if reference type fields are not on the device already we must copy them to the device and overwrite these fields with
            // pointers to the device memory.
            CopyReferenceTypeFieldsToDevice<T>(gpu, hostObject);

            var deviceTypeInfo = CreateDeviceType(newTypeToCreate);

            var deviceObject = Activator.CreateInstance(deviceTypeInfo.DeviceType);

            // For each element of the array, we need to do this:
            AssignPointerFields(gpu, hostObject, deviceObject, deviceTypeInfo.PointerFields);
            AssignNonPointerFields(hostObject, deviceObject, deviceTypeInfo.NonPointerFields);

            var deviceObjectArray = Array.CreateInstance(deviceTypeInfo.DeviceType, 1);
            deviceObjectArray.SetValue(deviceObject, 0);
            var copy1DArrayToDevice = CopyArrayToDevice(deviceTypeInfo.DeviceType, 1);
            object deviceObjectArrayPointer = copy1DArrayToDevice.Invoke(gpu, new object[] { deviceObjectArray });

            MapDeviceMemoryToObject(gpu, deviceObjectArrayPointer, hostObject);

            // this is not really an array, but it is a reference type, so make 0D to avoid addition of array length in kernel call.
            DevicePtrEx deviceObjectPointer = gpu.TryGetDeviceMemory(deviceObjectArrayPointer);
            typeof(DevicePtrEx).GetProperty("Dimensions").SetValue(deviceObjectPointer, 0, null);
            return deviceObjectArrayPointer;
        }

        /// <summary>
        /// Update any array fields of an object where the arrays are on the device. 
        /// </summary>
        public static void UpdateFromDevice<T>(GPGPU gpu, T hostObject)
        {
            if (hostObject.GetType().IsArray && hostObject.GetType().GetElementType().IsValueType)
                UpdateArrayFromDevice(gpu, hostObject as Array);
            // TODO? add recursion? Depends on how this is to be used.
            //foreach (var field in typeof(T).GetFields())
            foreach (var field in GetFieldsStandardLayout(typeof(T)))
            {
                if (field.FieldType.IsArray)
                {
                    UpdateArrayFromDevice(gpu, field.GetValue(hostObject) as Array);
                }
            }
        }

        #region PrivateStuff

        /// <summary>
        /// Copies any reference type fields (e.g. arrays) of the object to the device.
        /// </summary>
        private static void CopyReferenceTypeFieldsToDevice<T>(GPGPU gpu, T hostObject)
        {
            var fields = DeviceClassHelper.GetFieldsStandardLayout(hostObject.GetType());
            foreach (FieldInfo field in fields)
            {
                object fieldValue = field.GetValue(hostObject);
                // Ignore if CudafyIgnore
                if (field.GetCustomAttributes(typeof(CudafyIgnoreAttribute), true).Count() > 0) continue;
                // Only copy field to the device if this is not already done.
                if (field.FieldType.IsArray && !deviceObjectFromHostObject[gpu].ContainsKey(fieldValue))
                {
                    // If the elements of the array are value types, then we make a deep copy. Otherwise, we create an array of
                    // pointers to the objects.
                    if (field.FieldType.GetElementType().IsValueType)
                    {
                        Array hostArray = (Array)fieldValue;
                        CopyArrayToDevice(gpu, hostArray);
                    }
                    else
                    {
                        CopyArrayOfReferenceTypeToDevice(gpu, (Array)fieldValue);
                    }
                }
                else if (!field.FieldType.IsValueType && !deviceObjectFromHostObject[gpu].ContainsKey(fieldValue))
                {
                    CreateDeviceObject(gpu, fieldValue);
                }
            }
        }

        private static object CopyArrayOfReferenceTypeToDevice(GPGPU gpu, Array hostArray)
        {
            // The objects themselves must be copied to the device, filling up the array with a pointer to be object
            IntPtr[] pointerArray = new IntPtr[hostArray.Length];
            var arrayEnumerator = hostArray.GetEnumerator();
            int index = 0;
            while (arrayEnumerator.MoveNext())
            {
                CreateDeviceObject(gpu, arrayEnumerator.Current);
                var devicePointer = TryGetDeviceMemoryFromHostObject(gpu, arrayEnumerator.Current);
                pointerArray[index] = devicePointer.Pointer;
                index++;
            }
            // TODO add array of pointers, for field
            var copyArrayToDevice = CopyArrayToDevice(typeof(IntPtr), 1);
            Array deviceArray = copyArrayToDevice.Invoke(gpu, new object[] { pointerArray }) as Array;
            MapDeviceMemoryToObject(gpu, deviceArray, hostArray);
            return deviceArray;
        }

        private static object CopyArrayToDevice(GPGPU gpu, Array hostArray)
        {
            int dimension = hostArray.Rank;
            var copyArrayToDevice = CopyArrayToDevice(hostArray.GetType().GetElementType(), dimension);
            Array deviceArray = copyArrayToDevice.Invoke(gpu, new object[] { hostArray }) as Array;
            MapDeviceMemoryToObject(gpu, deviceArray, hostArray);
            return deviceArray;
        }

        private static void UpdateArrayFromDevice(GPGPU gpu, Array hostArray)
        {
            var devicePointer = TryGetDeviceMemoryFromHostObject(gpu, hostArray);
            if (devicePointer != null)
            {
                var copyArrayFromDevice = CopyArrayFromDevice(hostArray.GetType().GetElementType(), hostArray.Rank);
                var deviceArray = TryGetDeviceObjectFromHostObject(gpu, hostArray);
                copyArrayFromDevice.Invoke(gpu, new object[] { deviceArray, hostArray });
            }
        }

        private static void MapDeviceMemoryToObject(GPGPU gpu, object cudafyHandle, object newHandle)
        {
            deviceObjectFromHostObject[gpu].Add(newHandle, cudafyHandle);
        }

        /// <summary>
        /// Create type that contains pointers to device arrays.
        /// </summary>
        private static DeviceTypeInfo CreateDeviceType(Type hostType)
        {
            DeviceTypeInfo dummy;
            if (deviceTypeLookup.TryGetValue(hostType, out dummy))
                return dummy;

            // We are assuming that the device class is standard layout.
            // We further assume that all array fields are actually pointers to device memory.
            // For now assume that anything that is not an array is blittable.

            TypeBuilder tb = moduleBuilder.DefineType("DynamicType" + hostType.Name, TypeAttributes.Public |
                TypeAttributes.Sealed | TypeAttributes.SequentialLayout |
                TypeAttributes.Serializable, typeof(ValueType));

            var fields = GetFieldsStandardLayout(hostType);

            var pointerFields = new List<FieldMapping>();
            var nonPointerFields = new List<FieldMapping>();

            // Any class (i.e. not value type), which includes arrays, is replaced by an IntPtr
            foreach (var field in fields)
            {
                if (field.FieldType.IsClass)
                {
                    tb.DefineField(field.Name, typeof(IntPtr), FieldAttributes.Public);
                    FieldMapping newPointerFieldMapping;
                    // If this is an array, then we add in int fields to contain the dimensions of the array.
                    if (field.FieldType.IsArray)
                    {
                        newPointerFieldMapping = new ArrayFieldMapping() { Name = field.Name, HostObjectField = field };
                        ((ArrayFieldMapping)newPointerFieldMapping).ArrayRank = field.FieldType.GetArrayRank();
                        for (int r = 0; r < field.FieldType.GetArrayRank(); r++)
                        {
                            string dimensionFieldName = String.Format("{0}Len{1}", field.Name, r);
                            tb.DefineField(dimensionFieldName, typeof(int), FieldAttributes.Public);
                            ((ArrayFieldMapping)newPointerFieldMapping).ArrayDimensionNames.Add(dimensionFieldName);
                        }
                    }
                    else
                    {
                        newPointerFieldMapping = new FieldMapping() { Name = field.Name, HostObjectField = field };
                    }
                    pointerFields.Add(newPointerFieldMapping);
                }
                else
                {
                    tb.DefineField(field.Name, field.FieldType, FieldAttributes.Public);
                    nonPointerFields.Add(new FieldMapping() { Name = field.Name, HostObjectField = field });
                }
            }
            Type newType = tb.CreateType();
            foreach (var pointerField in pointerFields)
            {
                pointerField.DeviceObjectField = newType.GetFields().Where(f => f.Name == pointerField.Name).FirstOrDefault();
                if (pointerField is ArrayFieldMapping)
                {
                    foreach (string dimensionName in ((ArrayFieldMapping)pointerField).ArrayDimensionNames)
                    {
                        ((ArrayFieldMapping)pointerField).DeviceObjectDimensionFields.Add(
                            newType.GetFields().Where(f => f.Name == dimensionName).FirstOrDefault());
                    }
                }
            }
            foreach (var nonPointerField in nonPointerFields)
            {
                nonPointerField.DeviceObjectField = newType.GetFields().Where(f => f.Name == nonPointerField.Name).FirstOrDefault();
            }
            var deviceTypeInfo = new DeviceTypeInfo() { DeviceType = newType, NonPointerFields = nonPointerFields, PointerFields = pointerFields };
            deviceTypeLookup.Add(hostType, deviceTypeInfo);
            return deviceTypeInfo;
        }

        /// <summary>
        /// Get fields in the same order in which these would appear in memory for the device object (i.e. standard layout rules).
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        private static List<FieldInfo> GetFieldsStandardLayout(Type type)
        {
            return GetFieldsStandardLayout(new List<FieldInfo>(), type);
        }

        private static List<FieldInfo> GetFieldsStandardLayout(List<FieldInfo> fields, Type type)
        {
            if (type.BaseType == null) return fields;
            else
            {
                var newFields = type.GetFields(BindingFlags.DeclaredOnly | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                    .Where(f => f.GetCustomAttributes(typeof(CudafyIgnoreAttribute), true).Count() == 0);
                return GetFieldsStandardLayout(newFields.Concat(fields).ToList(), type.BaseType);
            }
        }

        private static void AssignNonPointerFields(object hostObject, object deviceObject, List<FieldMapping> nonPointerFields)
        {
            foreach (FieldMapping mapping in nonPointerFields)
            {
                object fieldValue = mapping.HostObjectField.GetValue(hostObject);
                mapping.DeviceObjectField.SetValue(deviceObject, fieldValue);
            }
        }

        private static void AssignPointerFields(GPGPU gpu, object hostObject, object deviceObject, List<FieldMapping> pointerFields)
        {
            foreach (FieldMapping mapping in pointerFields)
            {
                object fieldValue = mapping.HostObjectField.GetValue(hostObject);
                // Get the IntPtr to the device memory for the array.
                var devicePointer = TryGetDeviceMemoryFromHostObject(gpu, fieldValue);
                if (devicePointer == null) throw new ArgumentException("No device memory allocated for field " + mapping.Name);
                // The device object contains this pointer.
                mapping.DeviceObjectField.SetValue(deviceObject, devicePointer.Pointer);
                // If the field is an array then set the dimension fields too.
                if (mapping is ArrayFieldMapping)
                {
                    ArrayFieldMapping arrayFieldMapping = (ArrayFieldMapping)mapping;
                    Array array = fieldValue as Array;
                    for (int i = 0; i < arrayFieldMapping.ArrayRank; ++i)
                    {
                        arrayFieldMapping.DeviceObjectDimensionFields[i].SetValue(deviceObject, array.GetLength(i));
                    }
                }
            }
        }

        private static MethodInfo CopyArrayToDevice(Type arrayElementType, int arrayDimensions)
        {
            string arrayForm = ArrayFormFromDimensions(arrayDimensions);
            var methods = typeof(GPGPU).GetMethods().Where(t => t.Name == "CopyToDevice" && t.GetParameters().First().ParameterType.Name == arrayForm && t.GetParameters().Count() == 1);
            var method = methods.First();
            return method.MakeGenericMethod(new Type[] { arrayElementType });
        }

        private static MethodInfo CopyArrayFromDevice(Type arrayElementType, int arrayDimensions)
        {
            string arrayForm = ArrayFormFromDimensions(arrayDimensions);
            var methods = typeof(GPGPU).GetMethods().Where(t => t.Name == "CopyFromDevice" && t.GetParameters().First().ParameterType.Name == arrayForm && t.GetParameters().Last().ParameterType.Name == arrayForm);
            var method = methods.First();
            return method.MakeGenericMethod(new Type[] { arrayElementType });
        }

        private static string ArrayFormFromDimensions(int arrayDimensions)
        {
            switch (arrayDimensions)
            {
                case 1:
                    return "T[]";
                case 2:
                    return "T[,]";
                case 3:
                    return "T[,,]";
                default:
                    return string.Empty;
            }
        }

        #endregion
    }

    public class FieldMapping
    {
        public string Name { get; set; }
        public FieldInfo HostObjectField { get; set; }
        public FieldInfo DeviceObjectField { get; set; }
    }

    public class ArrayFieldMapping : FieldMapping
    {
        public int ArrayRank;
        public List<string> ArrayDimensionNames { get; set; }
        public List<FieldInfo> DeviceObjectDimensionFields { get; set; }

        public ArrayFieldMapping()
        {
            ArrayDimensionNames = new List<string>(); DeviceObjectDimensionFields = new List<FieldInfo>();
        }
    }
}
