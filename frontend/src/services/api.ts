import axios from 'axios'

export const TOKEN_KEY = 'wave_token'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
  timeout: 15_000,
})

/* Inject token on every request */
api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY)
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

/* Handle 401 — clear stale token */
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem(TOKEN_KEY)
      /* Let AuthContext react via its own isAuthenticated check */
    }
    return Promise.reject(err)
  }
)

export default api
