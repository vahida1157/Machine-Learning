	䃞ͪ???䃞ͪ???!䃞ͪ???	??S?2G&@??S?2G&@!??S?2G&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$䃞ͪ???&S????A?? ???YX9??v???*	433333K@2F
Iterator::Model?ݓ??Z??!_A@)?W[?????1??????;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9??v???!~<@)?]K?=??1?????r8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV-???!??????:@)M??St$??1??????4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?V-??!yxxxxPP@)????Mbp?1jiiiii@:Preprocessing2U
Iterator::Model::ParallelMapV2ŏ1w-!o?!??????@)ŏ1w-!o?1??????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C??6j?!??????@)-C??6j?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??b?!------@)/n??b?1------@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Mb??!jiiiii=@)?~j?t?X?1@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t18.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??S?2G&@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	&S????&S????!&S????      ??!       "      ??!       *      ??!       2	?? ????? ???!?? ???:      ??!       B      ??!       J	X9??v???X9??v???!X9??v???R      ??!       Z	X9??v???X9??v???!X9??v???JCPU_ONLYY??S?2G&@b 