	??HP????HP??!??HP??	?
O??:,@?
O??:,@!?
O??:,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??HP??ޓ??ZӬ?Ar?鷯??YǺ?????*	?????9P@2F
Iterator::Model??????!?0?L?G@)	?c???1?DB?CD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??d?`T??!?+?r??;@)ŏ1w-!??1???k7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?g??s???!?`?3U0@)	?^)ˀ?1???~E)@:Preprocessing2U
Iterator::Model::ParallelMapV2HP?s?r?!_?n?Y@)HP?s?r?1_?n?Y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?? ?rh??!|??y?1J@)????Mbp?1?hƁ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!?S̪?@)??_vOf?1?S̪?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*?c?!??h?@)a2U0*?c?1??h?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??@??ǈ?!?؊???2@)?~j?t?X?1??T?o}@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?
O??:,@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ޓ??ZӬ?ޓ??ZӬ?!ޓ??ZӬ?      ??!       "      ??!       *      ??!       2	r?鷯??r?鷯??!r?鷯??:      ??!       B      ??!       J	Ǻ?????Ǻ?????!Ǻ?????R      ??!       Z	Ǻ?????Ǻ?????!Ǻ?????JCPU_ONLYY?
O??:,@b 