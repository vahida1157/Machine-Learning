	Y?? ???Y?? ???!Y?? ???	?;Lx?#@?;Lx?#@!?;Lx?#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Y?? ???????B???A?[ A?c??Y/n????*	fffff&R@2F
Iterator::Model䃞ͪϕ?!???i?V=@)X?5?;N??1ӿЈ:G7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate	?c???!K?B#?B@)X9??v???1?N???Y5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"??u????!&!???7@)%u???1ˠT?x?4@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vO??!5_?g??-@)??_vO??15_?g??-@:Preprocessing2U
Iterator::Model::ParallelMapV2/n??r?!>???>@)/n??r?1>???>@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipNё\?C??!???eP?Q@)/n??r?1>???>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???B?i??!u?E]tD@)_?Q?k?1G????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zd?!??rW?@){?G?zd?1??rW?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t28.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?;Lx?#@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????B???????B???!????B???      ??!       "      ??!       *      ??!       2	?[ A?c???[ A?c??!?[ A?c??:      ??!       B      ??!       J	/n????/n????!/n????R      ??!       Z	/n????/n????!/n????JCPU_ONLYY?;Lx?#@b 