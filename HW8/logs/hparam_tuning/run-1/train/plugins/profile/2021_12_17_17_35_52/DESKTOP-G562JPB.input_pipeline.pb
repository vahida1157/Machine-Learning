	??1??%????1??%??!??1??%??	?N??d$@?N??d$@!?N??d$@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??1??%???^)???A,Ԛ????Y8gDio??*	gfffffH@2F
Iterator::ModelHP?sג?!??:?B@)?ZӼ???1G??).=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatŏ1w-!??!?h?>?%?@)-C??6??1???::@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea2U0*???!?Oq??3@)S?!?uq{?1?).?u+@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?4q?!?%C??6!@)?J?4q?1?%C??6!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipŏ1w-!??!?h?>?%O@)?q????o?1?!XG??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????g?!?h?>?@)?????g?1?h?>?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!?Oq??@)a2U0*?c?1?Oq??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM??St$??!?`m?'7@)_?Q?[?1K?`m?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t28.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?N??d$@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^)????^)???!?^)???      ??!       "      ??!       *      ??!       2	,Ԛ????,Ԛ????!,Ԛ????:      ??!       B      ??!       J	8gDio??8gDio??!8gDio??R      ??!       Z	8gDio??8gDio??!8gDio??JCPU_ONLYY?N??d$@b 