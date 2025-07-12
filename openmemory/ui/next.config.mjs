/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // 禁用尾部斜杠重定向，避免 308 状态码
  trailingSlash: false,
  // 添加API重写配置
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://mem0-openmemory-api:8765/api/v1/:path*',
      },
    ];
  },
}

export default nextConfig