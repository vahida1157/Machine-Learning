	????????????????!????????	6Ũ?oS$@6Ũ?oS$@!6Ũ?oS$@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????????????K7??A?߾?3??Y o?ŏ??*	??????K@2F
Iterator::Model??JY?8??!??<??zC@)?Q?????1?I?k?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? ?rh??!U?I?>@)??Pk?w??1w????8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;?O???!?"3u?4@)?St$????1o???q?-@:Preprocessing2U
Iterator::Model::ParallelMapV2?J?4q?!??L])@)?J?4q?1??L])@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?? ?rh??!U?I?N@)ŏ1w-!o?1a???I@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C??6j?!?????@)-C??6j?1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!Az?2C@)a??+ei?1Az?2C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u???!o??Nɲ7@)_?Q?[?1D?ܠj@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t29.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.96Ũ?oS$@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????K7??????K7??!????K7??      ??!       "      ??!       *      ??!       2	?߾?3???߾?3??!?߾?3??:      ??!       B      ??!       J	 o?ŏ?? o?ŏ??! o?ŏ??R      ??!       Z	 o?ŏ?? o?ŏ??! o?ŏ??JCPU_ONLYY6Ũ?oS$@b 