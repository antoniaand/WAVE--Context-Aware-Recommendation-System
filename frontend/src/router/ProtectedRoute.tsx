import { Navigate, Outlet } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'
import type { Role } from '@/types'

interface Props {
  allowedRoles?: Role[]
}

export function ProtectedRoute({ allowedRoles }: Props) {
  const { isAuthenticated, isLoading, role } = useAuth()

  if (isLoading) {
    return (
      <div style={{ minHeight: '100dvh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span className="btn-spinner" style={{ borderColor: 'var(--border-input)', borderTopColor: 'var(--accent)' }} />
      </div>
    )
  }

  if (!isAuthenticated) return <Navigate to="/login" replace />

  if (allowedRoles && !allowedRoles.includes(role)) {
    return <Navigate to="/home" replace />
  }

  return <Outlet />
}
