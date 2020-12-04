#include "darknetTR.h"
#define PY_SSIZE_T_CLEAN
#include "python3.7/Python.h"
#include <stdio.h>
bool gRun;
bool SAVE_RESULT = true;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

extern "C"
{


void copy_image_from_bytes(image im, unsigned char *pdata)
{
//    unsigned char *data = (unsigned char*)pdata;
//    int i, k, j;
    int w = im.w;
    int h = im.h;
    int c = im.c;
//    for (k = 0; k < c; ++k) {
//        for (j = 0; j < h; ++j) {
//            for (i = 0; i < w; ++i) {
//                int dst_index = i + w * j + w * h*k;
//                int src_index = k + c * i + c * w*j;
//                im.data[dst_index] = (float)data[src_index] / 255.;
//            }
//        }
//    }
    memcpy(im.data, pdata, h * w * c);

}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)xcalloc(h * w * c, sizeof(float));
    return out;
}

tk::dnn::Yolo3Detection* load_network(char* net_cfg, int n_classes, int n_batch)
{
    std::string net;
    net = net_cfg;
    tk::dnn::Yolo3Detection *detNN = new tk::dnn::Yolo3Detection;
    detNN->init(net, n_classes, 1);

    return detNN;
}
#include <typeinfo>
void do_inference(tk::dnn::Yolo3Detection *net, image im)
{
    std::vector<cv::Mat> batch_dnn_input;
    cv::Mat frame(im.h, im.w, CV_8UC3, (unsigned char*)im.data);
    batch_dnn_input.push_back(frame);
    net->update(batch_dnn_input, 1);

}

long cross_multiplication(float nom1, long nom2, long denom2) { return (long) ((nom1 * denom2)/nom2); }

PyObject* get_network_boxes(tk::dnn::Yolo3Detection *net, float thresh, int batch_num, PyObject* input_size, PyObject* resize_mesures) {

    std::vector<std::vector<tk::dnn::box>> batchDetected;
    batchDetected = net->get_batch_detected();

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyListObject *mask = (PyListObject*) PyList_New((Py_ssize_t) 0L);
    PyListObject *no_mask = (PyListObject*) PyList_New((Py_ssize_t) 0L);
    PyListObject *person = (PyListObject*) PyList_New((Py_ssize_t) 0L);
    PyListObject *bbox = (PyListObject*) PyList_New((Py_ssize_t) 4L);
    long x, y, w, h;

    for (int i = 0; i < batchDetected[batch_num].size(); ++i) {

        if (batchDetected[batch_num][i].prob > thresh) {

            // Scale boxes to frame size
            x = cross_multiplication(batchDetected[batch_num][i].x, PyLong_AsLong(input_size), PyLong_AsLong(PyList_GetItem(resize_mesures, (Py_ssize_t) 1L)));
            y = cross_multiplication(batchDetected[batch_num][i].y, PyLong_AsLong(input_size), PyLong_AsLong(PyList_GetItem(resize_mesures, (Py_ssize_t) 0L)));
            w = cross_multiplication(batchDetected[batch_num][i].w, PyLong_AsLong(input_size), PyLong_AsLong(PyList_GetItem(resize_mesures, (Py_ssize_t) 1L)));
            h = cross_multiplication(batchDetected[batch_num][i].h, PyLong_AsLong(input_size), PyLong_AsLong(PyList_GetItem(resize_mesures, (Py_ssize_t) 0L)));

            // Build bbox
            PyList_SetItem((PyObject*) bbox, 0L,(PyObject*) PyLong_FromLong(x));
            PyList_SetItem((PyObject*) bbox, 1L,(PyObject*) PyLong_FromLong(y));
            PyList_SetItem((PyObject*) bbox, 2L,(PyObject*) PyLong_FromLong(w));
            PyList_SetItem((PyObject*) bbox, 3L,(PyObject*) PyLong_FromLong(h));

            // Add the bbox to the appropriate list
            if (batchDetected[batch_num][i].cl == 2) {
                PyList_Append((PyObject*) person, (PyObject*) bbox);
            } else if (batchDetected[batch_num][i].cl == 1) {
                PyList_Append((PyObject*) mask, (PyObject*) bbox);
            } else if (batchDetected[batch_num][i].cl == 0) {
                PyList_Append((PyObject*) no_mask, (PyObject*) bbox);
            }

        }
    }
    // Dict with predictions
    PyDictObject* final_dets = (PyDictObject*) PyDict_New();
    PyDict_SetItem((PyObject*) final_dets, (PyObject*) PyUnicode_FromString("no_mask\0"), (PyObject*) no_mask);
    PyDict_SetItem((PyObject*) final_dets, (PyObject*) PyUnicode_FromString("face_mask\0"),(PyObject*) mask);
    PyDict_SetItem((PyObject*) final_dets, (PyObject*) PyUnicode_FromString("person\0"), (PyObject*) person);
    PyGILState_Release(gstate);
    return (PyObject*) final_dets;
}


//detection* get_network_boxes(tk::dnn::Yolo3Detection *net, float thresh, int batch_num, int *pnum)
//{
//    std::vector<std::vector<tk::dnn::box>> batchDetected;
//    batchDetected = net->get_batch_detected();
//    int nboxes =0;
//    std::vector<std::string> classesName = net->get_classesName();
//    detection* dets = (detection*)xcalloc(batchDetected[batch_num].size(), sizeof(detection));
//    for (int i = 0; i < batchDetected[batch_num].size(); ++i)
//    {
//        if (batchDetected[batch_num][i].prob > thresh)
//        {
//            dets[nboxes].cl = batchDetected[batch_num][i].cl;
//            strcpy(dets[nboxes].name, classesName[dets[nboxes].cl].c_str());
//            dets[nboxes].bbox.x = batchDetected[batch_num][i].x;
//            dets[nboxes].bbox.y = batchDetected[batch_num][i].y;
//            dets[nboxes].bbox.w = batchDetected[batch_num][i].w;
//            dets[nboxes].bbox.h = batchDetected[batch_num][i].h;
//            dets[nboxes].prob = batchDetected[batch_num][i].prob;
//            nboxes += 1;
//        }
//    }
//    if (pnum) *pnum = nboxes;
//    return dets;
//}

void main_loop(char *input, int n_batch) {
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::DetectionNN *detNN;
    detNN = &yolo;

    float conf_thresh = 0.3;
    std::string net = "/home/alex/Projects/jd/yolo4_int8.rt";
    detNN->init(net, 3, n_batch, conf_thresh);

    cv::VideoCapture cap(input); //input video

    cv::VideoWriter resultVideo;
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h)); //output video

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    cv::Mat frame;

    while(1) {
        batch_dnn_input.clear();
        batch_frame.clear();

        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame;
            if(!frame.data)
                break;

            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        }
        if(!frame.data)
            break;

        //inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw(batch_frame);

//        if(show){
//            for(int bi=0; bi< n_batch; ++bi){
//                cv::imshow("detection", batch_frame[bi]);
//                cv::waitKey(1);
//            }
//        }
        if(SAVE_RESULT){
            for(int i = 0; i < n_batch; i++){
                resultVideo << batch_frame[i];
            }
        }
    }
    double mean = 0;

    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;

}
}



