// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Model__Base__h__
#define __PUJ_ML__Model__Base__h__

#include <iostream>
#include <vector>
#include <Eigen/Core>

namespace PUJ_ML
{
  namespace Model
  {
    /**
     */
    template< class _R >
    class Base
    {
    public:
      using Self = Base;
      using TReal = _R;
      using TMatrix = Eigen::Matrix< _R, Eigen::Dynamic, Eigen::Dynamic >;
      using TCol = Eigen::Matrix< _R, Eigen::Dynamic, 1 >;
      using TRow = Eigen::Matrix< _R, 1, Eigen::Dynamic >;

    public:
      Base( const unsigned long long& n = 1 );
        virtual ~Base( ) = default;

      virtual unsigned long long number_of_parameters( ) const;
      virtual void set_number_of_parameters( const unsigned long long& n );

      virtual _R& operator()( const unsigned long long& i );
      virtual const _R& operator()( const unsigned long long& i ) const;

      virtual _R& operator()(
        const unsigned long long& i,
        const unsigned long long& j
        );
      virtual const _R& operator()(
        const unsigned long long& i,
        const unsigned long long& j
        ) const;

      template< class _Y, class _X >
      void evaluate(
        Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
        ) const;

      template< class _Y, class _X >
      void threshold(
        Eigen::EigenBase< _Y >& Y, const Eigen::EigenBase< _X >& X
        ) const;

    protected:
      virtual void _to_stream( std::ostream& o ) const;

    protected:
      std::vector< _R > m_P;

    public:
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._to_stream( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Model/Base.hxx>

#endif // __PUJ_ML__Model__Base__h__

// eof - $RCSfile$
