	???ZӼ?????ZӼ??!???ZӼ??		???&@	???&@!	???&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???ZӼ??0?'???A??&???Y??B?iޡ?*	     H@2F
Iterator::Model{?G?z??!VUUUU?D@)???Q???1     @?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatvq?-??!UUUUUu@@)F%u???1    ?;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?q?????!     @0@)??_vOv?1     ?&@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?zt?!VUUUU?$@){?G?zt?1VUUUU?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy?&1???!?????*M@)-C??6j?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!??????@)??_?Le?1??????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*?c?!      @)a2U0*?c?1      @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;?O??n??!     ?2@)a2U0*?S?1      @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t28.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9	???&@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0?'???0?'???!0?'???      ??!       "      ??!       *      ??!       2	??&?????&???!??&???:      ??!       B      ??!       J	??B?iޡ???B?iޡ?!??B?iޡ?R      ??!       Z	??B?iޡ???B?iޡ?!??B?iޡ?JCPU_ONLYY	???&@b 