import axios from "axios";
const API_URL = "http://localhost:8000/faq/";

export const getAnswerDefault = (question) => {
  return axios.post(API_URL + "default", question).then((response) => {
    return response.data;
  });
};

export const getAnswerCustomized = (question) => {
  return axios.post(API_URL + "customized", question).then((response) => {
    return response.data;
  });
};

export const getAnswerSimilarity = (question) => {
  return axios.post(API_URL + "similarity", question).then((response) => {
    return response.data;
  });
};

export const getAnswerTableQA = (question) => {
  return axios.post(API_URL + "table_qa", question).then((response) => {
    return response.data;
  });
};

export const getAnswerGPT2 = (question) => {
  return axios.post(API_URL + "gpt2", question).then((response) => {
    return response.data;
  });
};

export const getAnswerBloom = (question) => {
  return axios.post(API_URL + "bloom", question).then((response) => {
    return response.data;
  });
};
