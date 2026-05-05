import { createContext, useEffect, useState, useCallback, type ReactNode } from 'react'
import { authService } from '@/services/authService'
import { TOKEN_KEY } from '@/services/api'
import type { UserProfileResponse, UserProfile, Role } from '@/types'

interface AuthState {
  user: UserProfileResponse | null
  role: Role
  isAuthenticated: boolean
  isLoading: boolean
  hasProfile: boolean
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  updateProfile: (profile: UserProfile) => Promise<void>
  logout: () => void
  refreshUser: () => Promise<void>
}

export const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    role: 'user',
    isAuthenticated: false,
    isLoading: true,
    hasProfile: false,
  })

  const refreshUser = useCallback(async () => {
    try {
      const user = await authService.getMe()
      setState({
        user,
        role: user.role ?? 'user',
        isAuthenticated: true,
        isLoading: false,
        hasProfile: user.profile !== null,
      })
    } catch {
      localStorage.removeItem(TOKEN_KEY)
      setState({ user: null, role: 'user', isAuthenticated: false, isLoading: false, hasProfile: false })
    }
  }, [])

  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY)
    if (token) {
      refreshUser()
    } else {
      setState(s => ({ ...s, isLoading: false }))
    }
  }, [refreshUser])

  const login = useCallback(async (email: string, password: string) => {
    await authService.login(email, password)
    await refreshUser()
  }, [refreshUser])

  const register = useCallback(async (email: string, password: string) => {
    await authService.register(email, password)
    await refreshUser()
  }, [refreshUser])

  const updateProfile = useCallback(async (profile: UserProfile) => {
    await authService.updateProfile(profile)
    await refreshUser()
  }, [refreshUser])

  const logout = useCallback(() => {
    authService.logout()
    setState({ user: null, role: 'user', isAuthenticated: false, isLoading: false, hasProfile: false })
  }, [])

  return (
    <AuthContext.Provider value={{ ...state, login, register, updateProfile, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  )
}
