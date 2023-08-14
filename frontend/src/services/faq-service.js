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
