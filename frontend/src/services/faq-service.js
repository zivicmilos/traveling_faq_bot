import axios from "axios";
const API_URL = "http://localhost:8000/faq/";

export const getAnswer = (question) => {
  return axios.post(API_URL + "questions", question).then((response) => {
    return response.data;
  });
};
