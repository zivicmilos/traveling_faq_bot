import axios from "axios";
const API_URL = 'http://localhost:8000/faq/';

export const getAnswer = (question) => {
    console.log("question ", question);
        return (dispatch) => {
            return axios.post(API_URL+'questions/', question)
                .then((res) => {
                    console.log("response ", res);
                    dispatch({ type: "answer", payload : res.data.result })
                });
        }
      }