	[B>?٬??[B>?٬??![B>?٬??	?z\??	#@?z\??	#@!?z\??	#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$[B>?٬???w??#???AyX?5?;??Y?rh??|??*	     @_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"?uq??!?G?zI@)?X?? ??1q=
ףpG@:Preprocessing2F
Iterator::Model?=yX???!?Q???@@)Q?|a??1???Q?<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??@??ǈ?!\???(\#@)"??u????1??Q??@:Preprocessing2U
Iterator::Model::ParallelMapV2-C??6z?!{?G?z@)-C??6z?1{?G?z@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip@a??+??!
ףp=?P@)?+e?Xw?1>
ףp=@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?^)?p?!>
ףp=
@)	?^)?p?1>
ףp=
@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!gfffff@)y?&1?l?1gfffff@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???QI??!?z?G?&@)/n??b?1)\???(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?z\??	#@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?w??#????w??#???!?w??#???      ??!       "      ??!       *      ??!       2	yX?5?;??yX?5?;??!yX?5?;??:      ??!       B      ??!       J	?rh??|???rh??|??!?rh??|??R      ??!       Z	?rh??|???rh??|??!?rh??|??JCPU_ONLYY?z\??	#@b 