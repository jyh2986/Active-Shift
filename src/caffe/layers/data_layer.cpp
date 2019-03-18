#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <cstdlib>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }

  shuffle_size_ = this->layer_param_.data_param().shuffle_size();

  if(shuffle_size_!=0)
  {
	  int n;
	  for (n = 0; n < shuffle_size_; n++) {
		key_set.push_back(cursor_->key());
		cursor_->Next();
		if (!cursor_->valid()) {
		  n++;
		  break;
		}
	  }
	  LOG_IF(INFO, n<shuffle_size_ && Caffe::root_solver())<< "Shuffle_size is greater than total number of elements. Using "<<n<<" as shuffle size.";

	  const unsigned int prefetch_rng_seed = caffe_rng_rand();
	  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	  ShuffleLMDB();

	  LOG(INFO) << "Solver"<< Caffe::solver_rank()<< " - Shuffling LMDB : Shuffle Size(" << shuffle_size_ << "), Base Seed("<<prefetch_rng_seed<<")";
  }


}

template <typename Dtype>
bool DataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void DataLayer<Dtype>::Next() {
  if (this->layer_param_.phase() == TEST || shuffle_size_==0) {
	cursor_->Next();
	if (!cursor_->valid()) {
	  LOG_IF(INFO, Caffe::root_solver())
			  << "Restarting data prefetching from start.";
	  cursor_->SeekToFirst();
	}
  } else {
	iterator_++;

	if (iterator_ == key_set.end()) {
	  LOG_IF(INFO, Caffe::root_solver())
				<< "Restarting data prefetching from start.";
 	  ShuffleLMDB();
	}
  }
  offset_++;
}

template<typename Dtype>
void DataLayer<Dtype>::ShuffleLMDB() {
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(key_set.begin(), key_set.end(), prefetch_rng);
  iterator_ = key_set.begin();
}

//#define LMDB_DEBUG 1

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  Datum datum;
  string new_key;

#ifdef LMDB_DEBUG
  string temp_buf="";
#endif

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    if (shuffle_size_!=0) {
   	  string new_key = *iterator_;
	  cursor_->Retrieval(&new_key);

#ifdef LMDB_DEBUG
	  temp_buf+=(new_key+", ");
#endif
    }

    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      Dtype* top_label = batch->label_.mutable_cpu_data();
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();
    Next();
  }

#ifdef LMDB_DEBUG
  if(temp_buf!="") LOG(INFO)<<"[[[Solver"<<Caffe::solver_rank()<<"/Data Seq : "<<temp_buf<<"]]]";
#endif

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
