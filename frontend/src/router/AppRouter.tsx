import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { ProtectedRoute } from './ProtectedRoute'
import { LandingPage }    from '@/pages/LandingPage'
import { LoginPage }      from '@/pages/LoginPage'
import { RegisterPage }   from '@/pages/RegisterPage'
import { OnboardingPage } from '@/pages/OnboardingPage'
import { HomePage }       from '@/pages/HomePage'
import { ProfilePage }    from '@/pages/ProfilePage'
import { AdminPage }      from '@/pages/AdminPage'
import { ManagerPage }    from '@/pages/ManagerPage'
import { NotFoundPage }   from '@/pages/NotFoundPage'

function AnimatedRoutes() {
  const location = useLocation()
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <Routes location={location} key={location.pathname}>
        {/* Public */}
        <Route path="/"         element={<LandingPage />} />
        <Route path="/login"    element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* Authenticated (any role) */}
        <Route element={<ProtectedRoute />}>
          <Route path="/onboarding" element={<OnboardingPage />} />
          <Route path="/home"       element={<HomePage />} />
          <Route path="/profile"    element={<ProfilePage />} />
        </Route>

        {/* Admin only */}
        <Route element={<ProtectedRoute allowedRoles={['admin']} />}>
          <Route path="/admin" element={<AdminPage />} />
        </Route>

        {/* Event manager + admin */}
        <Route element={<ProtectedRoute allowedRoles={['event_manager', 'admin']} />}>
          <Route path="/manager" element={<ManagerPage />} />
        </Route>

        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </AnimatePresence>
  )
}

export function AppRouter() {
  return (
    <BrowserRouter>
      <AnimatedRoutes />
    </BrowserRouter>
  )
}
