import { atom } from "recoil";

export const session = atom({
  key: "session",
  default: {
    accessToken: window.localStorage.getItem("accessToken"),
    refreshToken: window.localStorage.getItem("refreshToken"),
  },
});
