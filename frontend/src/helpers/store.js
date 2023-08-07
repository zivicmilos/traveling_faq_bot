import { create } from "zustand";
import { getData } from "./local-storage";

const initialState = () => getData("items") || [];

export const useStore = create((set) => ({
  question: "",
  items: initialState(),
  model: "custom",
  preprocessing: false,
  weight: true,

  handleReset: () => set({ items: [] }),
  addItem: (item) => set((state) => ({ items: [...state.items, item] })),
  setQuestion: (question) => set({ question }),
  setModel: (model) => set({ model }),
  setPreprocessing: (preprocessing) => set({ preprocessing }),
  setWeight: (weight) => set({ weight }),
}));
