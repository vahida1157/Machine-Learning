	#??~j???#??~j???!#??~j???	??[??"(@??[??"(@!??[??"(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$#??~j???.?!??u??A[????<??YM?J???*	33333P@2F
Iterator::ModelJ+???!No#C@)n????1?I??k{>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ݓ??Z??!???S?d=@)??ǘ????1? ?G?19@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???S㥋?! V?;??4@){?G?z??1g@ʬ/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8??d?`??!??????N@){?G?zt?1g@ʬ@:Preprocessing2U
Iterator::Model::ParallelMapV2n??t?!?I??k{@)n??t?1?I??k{@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!/Y`Z??@)y?&1?l?1/Y`Z??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!2??/??@)??_vOf?12??/??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???H??!|???S?8@)a2U0*?c?1???*?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t19.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??[??"(@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	.?!??u??.?!??u??!.?!??u??      ??!       "      ??!       *      ??!       2	[????<??[????<??![????<??:      ??!       B      ??!       J	M?J???M?J???!M?J???R      ??!       Z	M?J???M?J???!M?J???JCPU_ONLYY??[??"(@b 