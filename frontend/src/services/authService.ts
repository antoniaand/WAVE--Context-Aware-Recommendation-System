import api, { TOKEN_KEY } from './api'
import type { LoginRequest, RegisterRequest, TokenResponse, UserProfileResponse } from '@/types'

export const authService = {
  async login(email: string, password: string): Promise<TokenResponse> {
    const { data } = await api.post<TokenResponse>('/auth/login', { email, password } as LoginRequest)
    localStorage.setItem(TOKEN_KEY, data.access_token)
    return data
  },

  async register(email: string, password: string, profile: RegisterRequest['profile']): Promise<TokenResponse> {
    const payload: RegisterRequest = { email, password, profile }
    const { data } = await api.post<TokenResponse>('/auth/register', payload)
    localStorage.setItem(TOKEN_KEY, data.access_token)
    return data
  },

  async getMe(): Promise<UserProfileResponse> {
    const { data } = await api.get<UserProfileResponse>('/auth/me')
    return data
  },

  logout(): void {
    localStorage.removeItem(TOKEN_KEY)
    api.post('/auth/logout').catch(() => { /* fire-and-forget */ })
  },
}
